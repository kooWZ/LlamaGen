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
import wandb
import base64
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)

from autoregressive.sample.sample_c2i_lib import do_sample

def init_wandb(run_name, wandb_dir, wandb_project='LlamaGen', wandb_entity='koowz-FVL25'):
    wandb.login(
        key=base64.b64decode(
            "ZmU2N2E1NjJkOGM5NjhjMjE1ZmU3Zjc1NDM2Zjc4YzljYTVkZWVjNg=="
        ).decode("utf-8")
    )
    wandb.init(
        project=wandb_project,
        name=run_name,
        dir=wandb_dir,
        entity=wandb_entity,
    )

def do_sample_and_save(ckpt, rank, device, name, args):
    save_to_folder = args.output_dir
    os.makedirs(save_to_folder, exist_ok=True)
    npz_file = os.path.join(save_to_folder, f"{name}_samples.npz")
    png_dir = os.path.join(save_to_folder, f"{name}_png")
    do_sample(ckpt, args, rank, device, npz_file, save_png_dir=png_dir)
    dist.barrier()
    if rank == 0:
        print(f"Saved samples for {name} to {npz_file} and {png_dir}")

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
    if rank == 0:
        print(f"Using DDP with {dist.get_world_size()} processes. Start sampling...")
    dist.barrier()

    ckpt_path = getattr(args, "gpt_ckpt", None)
    if ckpt_path is None or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint '{ckpt_path}' not found.")
    basename = os.path.splitext(os.path.basename(ckpt_path))[0]
    do_sample_and_save(ckpt_path, rank, device, basename, args)
    dist.barrier()
    dist.destroy_process_group()

def to_abs(path):
    if os.path.isabs(path):
        return path
    return os.path.join(llamagen_path, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="autoregressive/train/configs/llamagen.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/llamagen-b-384/000-GPT-B/checkpoints/0162500.pt"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/llamagen_samples"
    )
    cmdargs = parser.parse_args()

    cmdargs.config = to_abs(cmdargs.config)
    cmdargs.ckpt = to_abs(cmdargs.ckpt)

    assert os.path.exists(cmdargs.config)
    assert os.path.exists(cmdargs.ckpt)

    with open(cmdargs.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)
        def __getattr__(self, name):
            print(f"Warning: {name} not found in Args, returning None")
            return None

    args = Args(config)
    args.gpt_ckpt = cmdargs.ckpt
    args.output_dir = cmdargs.output_dir
    main(args)
