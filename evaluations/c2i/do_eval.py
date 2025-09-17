import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist


class Args:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)


torch.serialization.add_safe_globals([Args])
import argparse
import sys
import json
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from autoregressive.models.gpt import GPT_models

from autoregressive.sample.sample_c2i_lib import do_sample
from evaluations.c2i.eval_lib import evaluate


def main(args):
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
    sample_info = (
        f"{args.latent_size}-latents-"
        + f"{args.decoder_timesteps}-steps"
        + f"{args.decoder_guidance_scale}-scale"
    )
    if rank == 0:
        print(f"Using DDP with {dist.get_world_size()} processes. Start sampling...")
        os.makedirs(os.path.join(args.save_to, sample_info), exist_ok=True)
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
    npz_file = os.path.join(args.save_to, sample_info, "samples.npz")
    do_sample(gpt_model, args, rank, device, npz_file)
    if rank == 0:
        print("Sampling done. Start evaluation...")
    dist.barrier()
    if rank == 0:
        eval_result = evaluate(npz_file)
        print(f"Eval result: {eval_result}")
        result_file = os.path.join(args.save_to, sample_info, "result.json")
        with open(result_file, "w") as f:
            json.dump(eval_result, f, indent=4)
    dist.barrier()
    dist.destroy_process_group()


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-to",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--gpt-ckpt",
        type=str,
        default="/root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/92_1163523.pt",
    )
    parser.add_argument(
        "--vq-ckpt",
        type=str,
        default="/root/kongly/ckpts/flextok_d12_d12_in1k/",
        help="ckpt path for vq model",
    )
    parser.add_argument(
        "--latent-size", type=int, default=256, help="latent size of vq model"
    )
    parser.add_argument(
        "--decoder-timesteps", type=int, default=25, help="number of decoder steps"
    )
    parser.add_argument(
        "--decoder-guidance-scale",
        type=float,
        default=15,
        help="decoder guidance scale",
    )
    parser.add_argument("--gpt-model", type=str, default="GPT-Mini")
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
        "--vocab_size",
        type=int,
        default=64000,
        help="codebook size for vector quantization",
    )
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--cfg-interval", type=float, default=-1)
    parser.add_argument("--eval-per-gpu-batch-size", type=int, default=32)
    parser.add_argument("--eval-num-fid-samples", type=int, default=50000)
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
        "--eval-gather-freq",
        type=int,
        default=10,
        help="gather samples every N iterations to reduce memory usage",
    )
    args = parser.parse_args()
    if args.save_to is None:
        args.save_to = args.gpt_ckpt.replace("checkpoints", "results").replace(
            ".pt", ""
        )
    return args


if __name__ == "__main__":
    args = get_parser()
    main(args)
