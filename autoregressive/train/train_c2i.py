# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import subprocess
from glob import glob
from copy import deepcopy
import os
import time
import inspect
import argparse
import yaml
import threading
from huggingface_hub import HfApi
import base64

try:
    import wandb
except ImportError:
    pass

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from utils.logger import create_logger, setup_wandb
from utils.distributed import init_distributed_mode
from utils.ema import update_ema, requires_grad
from utils.lr_scheduler import CosineAnnealingWarmupLR
from dataset.build import build_dataset
from autoregressive.models.gpt import GPT_models
from evaluations.c2i.eval_lib import evaluate
from autoregressive.sample.sample_c2i_lib import do_sample


#################################################################################
#                             Training Helper Functions                         #
#################################################################################
def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    logger.info(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    logger.info(f"using fused AdamW: {fused_available}")
    return optimizer

def _upload_to_hf(checkpoint_path, logger):
    try:
        api = HfApi()
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"checkpoints/{checkpoint_path.split('/')[-1]}",
            repo_id="lykong/ar_1d_tok",
            repo_type="model",
            token=base64.b64decode(
                "aGZfemRwT0Fjd2RJZ0ZyRlpjekRxbXJFS21GaWlUaHZFZUVDZQ=="
            ).decode("utf-8"),
        )
        logger.info(f"[HF Upload] Uploaded {checkpoint_path}")
    except Exception as e:
        logger.warning(f"[HF Upload] Failed to upload {checkpoint_path}: {e}")


def save_checkpoint(
    logger, model, optimizer, scheduler, ema, train_steps, epoch, checkpoint_dir, args
):
    logger.info(f"Saving checkpoint at step {train_steps} (epoch {epoch})...")
    if not args.no_compile:
        model_weight = model.module._orig_mod.state_dict()
    else:
        model_weight = model.module.state_dict()
    checkpoint = {
        "model": model_weight,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "steps": train_steps,
        "args": vars(args),
    }
    if args.ema:
        checkpoint["ema"] = ema.state_dict()
    checkpoint_path = f"{checkpoint_dir}/{epoch}_{train_steps:07d}.pt"
    torch.save(checkpoint, checkpoint_path)
    latest_file = os.path.abspath(f"{checkpoint_dir}/../../latest.txt")
    with open(latest_file, "w") as f:
        f.write(checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")
    if args.upload_to_hf:
        logger.info(f"Uploading checkpoint to Hugging Face in background...")
        threading.Thread(
            target=_upload_to_hf,
            args=(checkpoint_path, logger),
            daemon=True,
        ).start()


def do_eval(model, args, rank, device, epoch, checkpoint_dir):
    npz_path = f"{checkpoint_dir}/eval_samples_epoch{epoch}.npz"
    do_sample(model, args, rank, device, npz_path)
    if rank == 0:
        return evaluate(npz_path)
    else:
        return {}


def calculate_eta(step_time, total_steps_remaining):
    if step_time == 0:
        return 0, 0, 0
    eta_seconds = step_time * total_steps_remaining
    eta_minutes, eta_seconds = divmod(eta_seconds, 60)
    eta_hours, eta_minutes = divmod(eta_minutes, 60)
    return int(eta_hours), int(eta_minutes), int(eta_seconds)


#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    init_distributed_mode(args)
    assert (
        args.global_batch_size % dist.get_world_size() == 0
    ), f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    def check_results_dir(path):
        if os.path.exists(path):
            return path
        base_path = os.path.abspath(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
        )
        new_path = os.path.join(base_path, path)
        print("Using new results_dir path:", new_path)
        return new_path

    # Setup an experiment folder:
    results_dir = check_results_dir(args.results_dir)
    if rank == 0:
        os.makedirs(
            results_dir, exist_ok=True
        )  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{results_dir}/*"))
    model_string_name = args.gpt_model.replace(
        "/", "-"
    )  # e.g., GPT-XL/2 --> GPT-XL-2 (for naming folders)
    experiment_dir = f"{results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = (
        f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    )
    dist.barrier()
    if rank == 0:
        os.makedirs(checkpoint_dir, exist_ok=False)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    if args.auto_resume:
        latest_file = f"{results_dir}/latest.txt"
        if os.path.exists(latest_file):
            with open(latest_file, "r") as f:
                gpt_ckpt = f.read().strip()
                if os.path.exists(gpt_ckpt):
                    args.gpt_ckpt = gpt_ckpt
                    logger.info(f"Auto-resume activated. Found latest checkpoint: {args.gpt_ckpt}")
                else:
                    logger.info(f"Auto-resume activated but latest checkpoint {gpt_ckpt} not found.")
        else:
            logger.info(f"Auto-resume activated but no latest.txt found at {latest_file}.")

    logger.info(f"{args}")
    logger.info(
        f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}."
    )

    # Setup model
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    latent_size = args.latent_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        use_liger=args.use_liger,
        fp32_attention=args.fp32_attention,
    ).to(device)

    wandb_extra_config = {}

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_embedding_params = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ("embed" not in n and p.requires_grad)
    )
    logger.info(
        f"GPT Parameters: {total_params}, Non-embedding Parameters: {non_embedding_params}"
    )
    wandb_extra_config["total_parameters"] = total_params
    wandb_extra_config["trainable_parameters"] = trainable_params
    wandb_extra_config["non_embedding_parameters"] = non_embedding_params

    if args.ema:
        ema = deepcopy(model).to(
            device
        )  # Create an EMA of the model for use after training
        requires_grad(ema, False)
        ema_params = sum(p.numel() for p in ema.parameters())
        logger.info(f"EMA Parameters: {ema_params:,}")

        wandb_extra_config["ema_parameters"] = ema_params

    # Setup optimizer
    optimizer = creat_optimizer(
        model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger
    )

    # Setup data:
    dataset = build_dataset(args)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Setup learning rate scheduler
    total_steps = args.epochs * len(loader)

    # Calculate warmup steps from warmup epochs if available
    if args.warmup_epochs is not None:
        warmup_steps = args.warmup_epochs * len(loader)
        logger.info(
            f"Using warmup_epochs: {args.warmup_epochs}, calculated warmup_steps: {warmup_steps}"
        )
    else:
        warmup_steps = args.warmup_steps
        logger.info(f"Using warmup_steps directly: {warmup_steps}")

    scheduler = CosineAnnealingWarmupLR(
        optimizer,
        init_lr=args.init_lr,
        base_lr=args.lr,
        final_lr=args.final_lr,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    # Log dataset info to wandb
    wandb_extra_config["dataset_size"] = len(dataset)
    wandb_extra_config["batch_size_per_gpu"] = int(
        args.global_batch_size // dist.get_world_size()
    )
    wandb_extra_config["total_steps_per_epoch"] = len(loader)
    wandb_extra_config["total_training_steps"] = total_steps

    use_wandb = setup_wandb(
        args, experiment_dir if rank == 0 else None, logger, rank, wandb_extra_config
    )
    # Prepare models for training:
    if args.gpt_ckpt:
        gpt_ckpt = args.gpt_ckpt
        if not os.path.exists(gpt_ckpt):
            base_path = os.path.abspath(
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
            )
            gpt_ckpt = os.path.join(base_path, gpt_ckpt)
            assert os.path.exists(gpt_ckpt), f"gpt_ckpt {gpt_ckpt} does not exist"
        checkpoint = torch.load(gpt_ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if args.ema:
            ema.load_state_dict(
                checkpoint["ema"] if "ema" in checkpoint else checkpoint["model"]
            )
        optimizer.load_state_dict(checkpoint["optimizer"])

        # Load scheduler state if available
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        train_steps = (
            checkpoint["steps"]
            if "steps" in checkpoint
            else int(args.gpt_ckpt.split("/")[-1].split(".")[0].split("_")[-1])
        )
        start_epoch = int(train_steps / int(len(dataset) / args.global_batch_size))
        train_steps = int(start_epoch * int(len(dataset) / args.global_batch_size))
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
        logger.info(f"Initial state: steps={train_steps}, epochs={start_epoch}")
    else:
        train_steps = 0
        start_epoch = 0
        if args.ema:
            update_ema(
                ema, model, decay=0
            )  # Ensure EMA is initialized with synced weights

    if not args.no_compile:
        logger.info("compiling the model... (may take several minutes)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model = DDP(model.to(device), device_ids=[args.gpu])
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    if args.ema:
        ema.eval()  # EMA model should always be in eval mode

    ptdtype = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.mixed_precision
    ]
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler("cuda", enabled=(args.mixed_precision == "fp16"))

    if args.ckpt_every_epoch is not None:
        ckpt_every = args.ckpt_every_epoch * len(loader)
        logger.info(
            f"Using ckpt_every_epoch: {args.ckpt_every_epoch}, calculated ckpt_every: {ckpt_every}"
        )
    else:
        ckpt_every = args.ckpt_every_iter
        logger.info(f"Using ckpt_every directly: {ckpt_every}")

    if args.eval_every_epoch is not None:
        eval_every = args.eval_every_epoch * len(loader)
        logger.info(
            f"Using eval_every_epoch: {args.eval_every_epoch}, calculated eval_every: {eval_every}"
        )
    else:
        eval_every = args.eval_every_iter

    logger.info(f"Training for {args.epochs} epochs...")
    total_steps_remaining = args.epochs * len(loader) - train_steps

    # Initialize skip counter for gradient norm threshold
    skipped_updates = 0
    skipped_updates_recent_100_steps = [0] * 100  # Track skips in the last 100 steps
    grad_norm_threshold = getattr(
        args, "grad_norm_threshold", float("inf")
    )  # Default to inf (no skipping)
    logger.info(f"Gradient norm threshold: {grad_norm_threshold}")

    check_grad_norm_for_skip = args.check_grad_norm_for_skip
    last_step_time = 0
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        data_start_time = step_start_time = time.time()
        for x, y, _ in loader:
            data_time = time.time() - data_start_time
            x = x.to(device, non_blocking=True).int()
            y = y.to(device, non_blocking=True).long()
            z_indices = x.reshape(x.shape[0], -1)
            c_indices = y.reshape(-1)
            assert z_indices.shape[0] == c_indices.shape[0]

            with torch.amp.autocast("cuda", dtype=ptdtype):
                _, loss = model(
                    cond_idx=c_indices, idx=z_indices[:, :-1], targets=z_indices
                )
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()

            # Need to unscale gradients to get true gradient norm
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            )

            if check_grad_norm_for_skip:
                # Synchronize gradient norm across all GPUs for consistent skip decision
                grad_norm_tensor = torch.tensor(grad_norm, device=device)
                dist.all_reduce(
                    grad_norm_tensor, op=dist.ReduceOp.MAX
                )  # Use max grad norm across all GPUs
                max_grad_norm = grad_norm_tensor.item()

                # Check if maximum gradient norm across all GPUs exceeds threshold
                skip_update = max_grad_norm > grad_norm_threshold
            else:
                skip_update = False
            skipped_updates_recent_100_steps.pop(0)
            skipped_updates_recent_100_steps.append(1 if skip_update else 0)
            recent_skip_rate = sum(skipped_updates_recent_100_steps) / 100

            if not skip_update:
                # step the optimizer and scaler if training in fp16
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # Update the learning rate
                # flush the gradients as soon as we can, no need for this memory anymore
                optimizer.zero_grad(set_to_none=True)
                if args.ema:
                    update_ema(
                        ema,
                        model.module._orig_mod if not args.no_compile else model.module,
                    )
            else:
                # Skip the update and zero gradients
                skipped_updates += 1  # Increment on all GPUs to keep count synchronized
                if rank == 0:  # Only log from rank 0 to avoid duplicate messages
                    logger.info(
                        f"Skipped update due to large gradient norm: max_grad_norm={max_grad_norm:.4f} > {grad_norm_threshold} (local_grad_norm={grad_norm:.4f})"
                    )
                optimizer.zero_grad(set_to_none=True)
                # Update scaler for consistency but don't step optimizer
                scaler.update()
                scheduler.step()  # Update the learning rate

            # Total step time
            train_steps += 1
            total_steps_remaining -= 1
            extra_metrics = {}

            # Save checkpoint:
            if (
                ckpt_every is not None
                and train_steps % ckpt_every == 0
                and train_steps > 0
            ):
                if rank == 0:
                    checkpoint_start_time = time.time()
                    save_checkpoint(
                        logger,
                        model,
                        optimizer,
                        scheduler,
                        ema if args.ema else None,
                        train_steps,
                        epoch,
                        checkpoint_dir,
                        args,
                    )
                    ckpt_time = time.time() - checkpoint_start_time
                    extra_metrics["timing/save_checkpoint"] = ckpt_time
                dist.barrier()

            if (
                eval_every is not None
                and train_steps % eval_every == 0
                and train_steps > 0
            ):
                eval_start_time = time.time()
                model.eval()
                try:
                    eval_metrics = do_eval(
                        model, args, rank, device, epoch, checkpoint_dir
                    )
                    eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    eval_metrics["timing/eval_time_sec"] = time.time() - eval_start_time
                    extra_metrics.update(eval_metrics)
                except Exception as e:
                    logger.warning(f"Evaluation failed: {e}")
                dist.barrier()
                model.train()

            if train_steps % args.reduce_grad_norm_every_iter == 0:
                reduce_tensor = torch.tensor(
                    [loss.detach(), grad_norm],
                    device=device
                )
                reduce_req = dist.reduce(
                    reduce_tensor,
                    dst=0,
                    op=dist.ReduceOp.SUM,
                    async_op=True
                )
                reduce_req.wait()

                if rank == 0:
                    avg_step_loss, avg_grad_norm = (
                        reduce_tensor / dist.get_world_size()
                    ).tolist()
            else:
                if rank == 0:
                    avg_step_loss, avg_grad_norm = loss.item(), grad_norm

            # Log to console
            if rank == 0:
                eta_hours, eta_minutes, eta_seconds = calculate_eta(
                    last_step_time,
                    total_steps_remaining,
                )
                eta_msg = f"ETA: {eta_hours:02d}:{eta_minutes:02d}:{eta_seconds:02d}"
                log_msg = f"(step={train_steps:07d}) Train Step Loss: {avg_step_loss:.4f}"
                log_msg += f", Data Time: {data_time*1000:.1f}ms, Last Step Time: {last_step_time*1000:.1f}ms"
                log_msg += f", Grad Norm: {avg_grad_norm:.4f}"
                if skip_update:
                    log_msg += f", SKIPPED UPDATE (max_grad_norm={avg_grad_norm:.4f})"
                log_msg += f", {eta_msg}"
                logger.info(log_msg)

                # Log to wandb
                if use_wandb:
                    # Get current learning rate
                    current_lr = optimizer.param_groups[0]["lr"]
                    wandb_logs = {
                        "train/loss": avg_step_loss,
                        "train/epoch": epoch,
                        "train/step": train_steps,
                        "train/learning_rate": current_lr,
                        "timing/data_time_ms": data_time * 1000,
                        "timing/step_time_ms": last_step_time * 1000,
                        "train/grad_norm": avg_grad_norm,
                        "skipping/skipped_updates_total": skipped_updates,
                        "skipping/skip_rate": (
                            skipped_updates / train_steps if train_steps > 0 else 0
                        ),
                        "skipping/skip_rate_recent_100_steps": recent_skip_rate,
                    }
                    if skip_update:
                        wandb_logs["train/update_skipped"] = 1
                    else:
                        wandb_logs["train/update_skipped"] = 0
                    wandb_logs.update(extra_metrics)
                    wandb.log(wandb_logs, step=train_steps)

            curr_time = time.time()
            last_step_time = curr_time - step_start_time
            data_start_time = step_start_time = curr_time

    dist.barrier()
    if rank == 0:
        save_checkpoint(
            logger,
            model,
            optimizer,
            scheduler,
            ema if args.ema else None,
            train_steps,
            "final",
            checkpoint_dir,
            args,
        )
    if use_wandb and rank == 0:
        wandb.finish()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)
        def __getattr__(self, name):
            print(f"Warning: {name} not found in Args, returning None")
            return None

    main(Args(config))
