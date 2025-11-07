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

from autoregressive.sample.sample_c2i_lib import do_sample
from evaluations.c2i.torch_eval import evaluate

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
        os.makedirs(args.save_to_folder, exist_ok=True)

    npz_file = os.path.join(args.save_to_folder, "samples.npz")
    do_sample(args.gpt_ckpt, args, rank, device, npz_file)
    if rank == 0:
        print("Sampling done. Start evaluation...")
    dist.barrier()
    if rank == 0:
        eval_result = evaluate(npz_file)
        try:
            os.remove(npz_file)
        except:
            pass
        print(f"Eval result: {eval_result}")
        result_file = os.path.join(args.save_to_folder, "result.json")
        with open(result_file, "w") as f:
            json.dump(eval_result, f, indent=4)
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
        default="configs/ours_0250000_b.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True
    )
    parser.add_argument(
        "--save-to-folder",
        type=str,
        required=True
    )
    cmdargs = parser.parse_args([])

    cmdargs.config = to_abs(cmdargs.config)
    cmdargs.ckpt = to_abs(cmdargs.ckpt)
    cmdargs.save_to_folder = to_abs(cmdargs.save_to_folder)

    assert os.path.exists(cmdargs.config)
    assert os.path.exists(cmdargs.ckpt)

    with open(cmdargs.config, "r") as file:
        import yaml
        config = yaml.load(file, Loader=yaml.FullLoader)

    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)
        def __getattr__(self, name):
            print(f"Warning: {name} not found in Args, returning None")
            return None

    args = Args(config)
    args.save_to_folder = cmdargs.save_to_folder
    args.gpt_ckpt = cmdargs.ckpt
    main(args)
