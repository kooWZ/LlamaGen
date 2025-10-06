# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/train.py
#   nanoGPT: https://github.com/karpathy/nanoGPT/blob/master/model.py
import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
import os
import time
import argparse
import yaml
import random
import numpy as np
import json
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from utils.logger import create_logger
from utils.distributed import init_distributed_mode
from evaluations.c2i.torch_eval import evaluate
from autoregressive.sample.sample_c2i_lib import do_sample_flextok, do_sample_titok


@torch.compiler.disable(recursive=True)
def do_eval(ckpt_path, args, rank, device, filename, checkpoint_dir, logger, remove_npz=False):
    npz_path = f"{checkpoint_dir}/eval_samples_{filename}.npz"
    if args.decoder_type == "flextok":
        do_sample_flextok(ckpt_path, args, rank, device, npz_path)
    elif args.decoder_type == "titok":
        do_sample_titok(ckpt_path, args, rank, device, npz_path)
    else:
        logger.warning(
            f"Unknown decoder_type {args.decoder_type}, skipping evaluation."
        )
        return {}
    if rank == 0:
        result = evaluate(npz_path)
        if remove_npz:
            os.remove(npz_path)
        logger.info(f"Eval results at {filename}: {result}")
        if args.save_eval_results_to is not None:
            with open(args.save_eval_results_to, "w") as f:
                json.dump(result, f, indent=4)
        return result
    else:
        return {}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_results_dir(path):
    if os.path.exists(path):
        return path
    base_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
    )
    new_path = os.path.join(base_path, path)
    print("Using new results_dir path:", new_path)
    return new_path


def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    init_distributed_mode(args)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    seed_everything(seed)
    torch.cuda.set_device(device)

    gpt_ckpt = check_results_dir(args.gpt_ckpt)
    assert os.path.exists(gpt_ckpt), f"gpt_ckpt {gpt_ckpt} does not exist."
    logger = create_logger(None)

    eval_args = f"gpt_ckpt={gpt_ckpt}, decoder={args.decoder_type}, latent_size={args.eval_latent_size}, ar_cfg_scale={args.cfg_scale}, ar_temperature={args.temperature}, ar_top_k={args.top_k}, ar_top_p={args.top_p}"
    if args.decoder_type == "flextok":
        eval_args += f", decoder_timesteps={args.decoder_timesteps}, decoder_guidance_scale={args.decoder_guidance_scale}"

    logger.info(f"Eval args: {eval_args}")

    eval_start_time = time.time()
    do_eval(
        gpt_ckpt,
        args,
        rank,
        device,
        args.eval_filename,
        os.path.dirname(gpt_ckpt),
        logger,
        remove_npz=True,
    )
    logger.info(f"Eval took {time.time() - eval_start_time} seconds.")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="configs/test.yaml",
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        help="Additional configuration options in the format key=value",
    )
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if args.override:
        for override_arg in args.override:
            try:
                key, value = override_arg.split("=", 1)
                if value.isdigit():
                    value = int(value)
                elif value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                config[key] = value
            except ValueError:
                print(
                    f"Invalid override argument: {override_arg}. Expected format is key=value."
                )
                exit(1)

    class Args:
        def __init__(self, config):
            for key, value in config.items():
                setattr(self, key, value)

        def __getattr__(self, name):
            print(f"Warning: {name} not found in Args, returning None")
            return None

    main(Args(config))
