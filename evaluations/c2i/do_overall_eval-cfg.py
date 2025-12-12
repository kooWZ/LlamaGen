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


def resolve_cfg_scales(args):
    """
    Allow cfg_scale to be either a single float/int or a list of values loaded from YAML.
    """
    values = getattr(args, "cfg_scale", None)
    if values is None:
        return [1.0]
    if isinstance(values, (list, tuple)):
        return [float(v) for v in values]
    if isinstance(values, str):
        try:
            return [float(values)]
        except ValueError:
            raise ValueError(f"cfg_scale string '{values}' cannot be parsed as float.")
    return [float(values)]

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
    save_to_folder = getattr(args, "eval_output_dir", None) or os.path.dirname(ckpt)
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
        print(f"Eval result: {eval_result}")
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

    ckpt_path = getattr(args, "gpt_ckpt", None) or getattr(args, "gpt_ckpts", None)
    if ckpt_path is None:
        raise ValueError("gpt_ckpt not provided in config/arguments.")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint path '{ckpt_path}' does not exist.")
    ckpt_basename = os.path.splitext(os.path.basename(ckpt_path))[0]
    save_dir = getattr(args, "eval_output_dir", None)
    if save_dir is None:
        save_dir = os.path.dirname(ckpt_path)
    os.makedirs(save_dir, exist_ok=True)
    args.eval_output_dir = save_dir
    args.gpt_ckpts = save_dir  # backwards compatibility for downstream calls

    # Setup DDP:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    if rank == 0:
        try:
            init_wandb(f"{ckpt_basename}_eval", args.eval_output_dir)
        except Exception as e:
            print(f"Error init wandb {e}")
        print(f"Using DDP with {dist.get_world_size()} processes. Start sampling...")
    dist.barrier()

    cfg_scales = resolve_cfg_scales(args)
    overall_data = {}
    for cfg_scale in cfg_scales:
        args.cfg_scale = cfg_scale
        scale_tag = str(cfg_scale).replace(".", "p")
        run_name = f"{ckpt_basename}_cfg{scale_tag}"
        res = do_eval(ckpt_path, rank, device, run_name, args)
        if rank == 0:
            try:
                log_payload = {f"eval/{k}": v for k, v in res.items()}
                wandb.log(log_payload)
            except Exception as e:
                print(f"Error upload to wandb {e}")
            try:
                with open(os.path.join(args.eval_output_dir, f"{run_name}_result.json"), "r") as f:
                    data = json.load(f)
                    overall_data[run_name] = data
            except Exception as e:
                print(f"Error read eval json {e}")
        dist.barrier()
    if rank == 0:
        with open(os.path.join(args.eval_output_dir, f"{ckpt_basename}_overall_result-cfg.json"), "w") as f:
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
        default="autoregressive/train/configs/final_b-cfg.yaml",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="outputs/final_semtok-1stage-B/000-GPT-B/checkpoints/259_0162500.pt"
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
    if not hasattr(args, "eval_output_dir") or args.eval_output_dir is None:
        args.eval_output_dir = os.path.dirname(cmdargs.ckpt)
    os.makedirs(args.eval_output_dir, exist_ok=True)
    args.gpt_ckpts = args.eval_output_dir
    main(args)
