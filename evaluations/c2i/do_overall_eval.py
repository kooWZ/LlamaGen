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
from evaluations.c2i.torch_eval import evaluate

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

def do_eval(ckpt, rank, device, name, args):
    save_to_folder = args.gpt_ckpts
    npz_file = os.path.join(save_to_folder, f"{name}_samples.npz")
    do_sample(ckpt, args, rank, device, npz_file)
    if rank == 0:
        print("Sampling done. Start evaluation...")
    dist.barrier()
    if rank == 0:
        eval_result = evaluate(npz_file)
        try:
            os.remove(npz_file)
        except:
            pass
        print(f"Ckpt {ckpt} Eval result: {eval_result}")
        result_file = os.path.join(save_to_folder, f"{name}_result.json")
        with open(result_file, "w") as f:
            json.dump(eval_result, f, indent=4)
    dist.barrier()
    if rank == 0:
        return eval_result
    else:
        return {}

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
        try:
            init_wandb(args.wandb_name, args.gpt_ckpts)
        except Exception as e:
            print(f"Error init wandb {e}")
        print(f"Using DDP with {dist.get_world_size()} processes. Start sampling...")
    dist.barrier()

    overall_data = {}
    for ckpt in sorted(list(os.listdir(args.gpt_ckpts))):
        if ckpt.endswith(".pt"):
            basename = ckpt.split(".")[0]
            epoch, step = basename.split("_")
            if not epoch.isdigit():
                continue
            epoch, step = int(epoch), int(step)
            res = do_eval(os.path.join(args.gpt_ckpts, ckpt), rank, device, basename, args)
            if rank == 0:
                try:
                    res = {f"eval/{k}": v for k, v in res.items()}
                    wandb.log(res, step=step)
                except Exception as e:
                    print(f"Error upload to wandb {e}")
                try:
                    with open(os.path.join(args.gpt_ckpts, f"{basename}_result.json"), "r") as f:
                        data = json.load(f)
                        overall_data[basename] = data
                except Exception as e:
                    print(f"Error read eval json {e}")
            dist.barrier()
    if rank == 0:
        with open(os.path.join(args.gpt_ckpts, f"overall_result.json"), "w") as f:
            json.dump(overall_data, f, indent=4)
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
        "--ckpts",
        type=str,
        default="outputs/llamagen-b-384/000-GPT-B/checkpoints"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default="llamagen"
    )
    cmdargs = parser.parse_args()

    cmdargs.config = to_abs(cmdargs.config)
    cmdargs.ckpts = to_abs(cmdargs.ckpts)

    assert os.path.exists(cmdargs.config)
    assert os.path.exists(cmdargs.ckpts)

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
    args.gpt_ckpts = cmdargs.ckpts
    main(args)
