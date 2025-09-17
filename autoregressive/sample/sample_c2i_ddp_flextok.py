# Modified from:
#   DiT:  https://github.com/facebookresearch/DiT/blob/main/sample_ddp.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from tqdm import tqdm
import os
import time
import numpy as np
import math
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

flextok_path = os.path.abspath(
    os.path.join(llamagen_path, "external_tokenizers/flextok")
)
sys.path.append(flextok_path)
from flextok.flextok_wrapper import FlexTokFromHub
from flextok.utils.demo import denormalize
from flextok.utils.misc import get_bf16_context, get_generator

class Args:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)
torch.serialization.add_safe_globals([Args])


def main(args):
    # Setup PyTorch:
    assert (
        torch.cuda.is_available()
    ), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # create and load model
    vq_model = FlexTokFromHub.from_pretrained(args.vq_ckpt)
    vq_model.to(device)
    vq_model.eval()

    # create and load gpt model
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]

    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=args.latent_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)
    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
    if args.from_fsdp:  # fsdp
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception(
            "please check model weight, maybe add --from-fsdp to run command"
        )
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model, mode="reduce-overhead", fullgraph=True
        )  # requires PyTorch 2.0 (optional)
    else:
        print(f"no model compile")

    # Create folder for NPZ output:
    ckpt_name = os.path.basename(args.gpt_ckpt).replace(".pt", "")
    folder_name = (
        f"{ckpt_name}-"
        f"topk-{args.top_k}-topp-{args.top_p}-temperature-{args.temperature}-"
        f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    )
    if args.sample_dir is None:
        args.sample_dir = os.path.join(os.path.dirname(args.gpt_ckpt), "samples")
    sample_folder_dir = os.path.join(args.sample_dir, args.folder_name, folder_name)
    if rank == 0:
        os.makedirs(args.sample_dir, exist_ok=True)
        print(f"Will save NPZ file as {sample_folder_dir}.npz")
    dist.barrier()

    # Figure out how many samples we need to generate on each GPU and how many iterations we need to run:
    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(
        math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size
    )
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar) if rank == 0 else pbar
    total = 0

    # Use CPU storage and periodic gathering to manage memory efficiently
    batch_samples = []  # Temporary storage for current batch
    batch_labels = []   # Temporary storage for corresponding class indices

    # rank0 maintains all collected samples
    if rank == 0:
        final_samples_list = []
        final_labels_list = []
        print(f"Will gather samples every {args.gather_freq} iterations to manage memory efficiently")

    # Store image shape once we know it
    img_shape = None

    for iteration in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)

        index_sample = generate(
            gpt_model,
            c_indices,
            args.latent_size,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        with get_bf16_context(True):
            reconst = vq_model.detokenize(
                [index_sample[t].unsqueeze(0) for t in range(index_sample.shape[0])],
                timesteps=args.decoder_timesteps,  # Number of denoising steps
                guidance_scale=args.decoder_guidance_scale,  # Classifier-free guidance scale
                perform_norm_guidance=True,  # APG, see https://arxiv.org/abs/2410.02416
                # Optionally control initial noise. Note that while the initial noise is deterministic, the rest of the model isn't.
                generator=get_generator(seed=0, device=device),
                verbose=False,  # Enable to show denoising progress bar with tqdm
            )

        start_time = time.time()
        # Convert tensors directly to CPU numpy arrays to save GPU memory
        batch_data = []
        batch_class_indices = []
        for i in range(n):
            # Convert tensor to numpy in the same format as the original PNG->numpy pipeline
            # Immediately move to CPU to save GPU memory
            img_tensor = denormalize(reconst[i].cpu()).clamp(0, 1)
            # Convert from tensor (C, H, W) to numpy (H, W, C) and to uint8
            img_numpy = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            batch_data.append(img_numpy)
            # Store corresponding class index
            batch_class_indices.append(c_indices[i].cpu().numpy())

        # Store in CPU memory and record image shape
        if batch_data and img_shape is None:
            img_shape = batch_data[0].shape
        batch_samples.extend(batch_data)
        batch_labels.extend(batch_class_indices)
        # print(f"Convert and store {len(batch_data)} images and labels took {time.time() - start_time} seconds")
        total += global_batch_size

        # Periodic gathering to rank0 to manage memory
        if (iteration + 1) % args.gather_freq == 0 or (iteration + 1) == iterations:
            start_gather_time = time.time()

            # Convert current batch to numpy arrays for gathering
            if batch_samples:
                rank_batch_images = np.stack(batch_samples)
                rank_batch_labels = np.array(batch_labels)
                batch_samples = []  # Clear after gathering
                batch_labels = []   # Clear after gathering
            else:
                # Create empty arrays with correct shape
                if img_shape is None:
                    img_shape = (256, 256, 3)  # Default fallback
                rank_batch_images = np.empty((0, *img_shape), dtype=np.uint8)
                rank_batch_labels = np.empty((0,), dtype=np.int64)

            if rank == 0:
                print(f"Gathering samples from iteration {iteration + 1}...")

            # Gather this batch from all ranks (both images and labels)
            gathered_images = [None for _ in range(dist.get_world_size())]
            gathered_labels = [None for _ in range(dist.get_world_size())]
            torch.distributed.all_gather_object(gathered_images, rank_batch_images)
            torch.distributed.all_gather_object(gathered_labels, rank_batch_labels)

            # rank0 processes and stores the gathered data
            if rank == 0:
                for rank_images, rank_labels in zip(gathered_images, gathered_labels):
                    if rank_images.shape[0] > 0:
                        final_samples_list.append(rank_images)
                        final_labels_list.append(rank_labels)

                total_collected = sum(arr.shape[0] for arr in final_samples_list)
                print(f"Rank0: collected {total_collected} total samples and labels so far")

            print(f"Rank {rank}: gather took {time.time() - start_gather_time} seconds")

            # Synchronize all processes
            dist.barrier()

    # Final processing and NPZ creation (data already collected via periodic gathering)
    dist.barrier()

    if rank == 0:
        if final_samples_list and final_labels_list:
            print("Concatenating all collected samples and labels...")
            final_samples = np.concatenate(final_samples_list, axis=0)
            final_labels = np.concatenate(final_labels_list, axis=0)

            # Truncate to exact number of samples needed (maintain correspondence)
            final_samples = final_samples[:args.num_fid_samples]
            final_labels = final_labels[:args.num_fid_samples]

            # Save directly to NPZ with both images and labels
            npz_path = f"{sample_folder_dir}.npz"
            np.savez(npz_path, arr_0=final_samples, labels=final_labels)
            print(f"Saved .npz file to {npz_path}")
            print(f"  Images shape: {final_samples.shape}")
            print(f"  Labels shape: {final_labels.shape}")
            print("Memory efficient generation with labels completed!")
        else:
            print("No samples collected!")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-Mini"
    )
    parser.add_argument(
        "--gpt-ckpt",
        type=str,
        default="/root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/92_1163523.pt",
    )
    parser.add_argument(
        "--gpt-type",
        type=str,
        choices=["c2i", "t2i"],
        default="c2i",
        help="class-conditional or text-conditional",
    )
    parser.add_argument("--from-fsdp", action="store_true")
    parser.add_argument(
        "--cls-token-num",
        type=int,
        default=1,
        help="max token number of condition input",
    )
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"]
    )
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument(
        "--vq-ckpt",
        type=str,
        default="/root/kongly/ckpts/flextok_d12_d12_in1k/",
        help="ckpt path for vq model",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=64000,
        help="codebook size for vector quantization",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--sample-dir", type=str, default=None)
    parser.add_argument("--per-proc-batch-size", type=int, default=64)
    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument(
        "--top-k", type=int, default=0, help="top-k value to sample with"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature value to sample with",
    )
    parser.add_argument(
        "--top-p", type=float, default=1.0, help="top-p value to sample with"
    )
    parser.add_argument(
        "--folder-name", type=str, default="debug", help="folder name to save samples"
    )
    parser.add_argument(
        "--decoder-timesteps", type=int, default=25, help="number of decoder steps"
    )
    parser.add_argument(
        "--decoder-guidance-scale", type=float, default=7.5, help="decoder guidance scale"
    )
    parser.add_argument(
        "--latent-size", type=int, default=256, help="latent size of vq model"
    )
    parser.add_argument(
        "--gather-freq", type=int, default=10, help="gather samples every N iterations to reduce memory usage"
    )
    args = parser.parse_args()
    main(args)
