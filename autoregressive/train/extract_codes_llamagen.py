# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import os
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import json

torch.set_grad_enabled(False)

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset
from autoregressive.sample.sample_c2i_lib import LlamaGenDecoder

from tqdm import tqdm
from glob import glob


def convert_path(path, force=False):
    if force:
        return os.path.join(llamagen_path, path)
    if os.path.exists(path) or os.path.isabs(path):
        return path
    else:
        return os.path.join(llamagen_path, path)

#################################################################################
#                                  Training Loop                                #
#################################################################################
def main(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # Setup DDP:
    if not args.single:
        init_distributed_mode(args)
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    else:
        device = "cuda"
        rank = 0

    # Setup a feature folder:
    if args.single or rank == 0:
        os.makedirs(args.code_path, exist_ok=True)
        os.makedirs(
            os.path.join(args.code_path, f"{args.dataset}{args.image_size}_codes"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(args.code_path, f"{args.dataset}{args.image_size}_labels"),
            exist_ok=True,
        )
    if not args.single:
        dist.barrier()
    done_files = glob(
        os.path.join(
            args.code_path, f"{args.dataset}{args.image_size}_codes", f"{rank}_*.npy"
        )
    )
    last_index = len(done_files) - 1
    print(f"Rank {rank} has done {len(done_files)} files, last index {last_index}.")

    # create and load model
    vq_model = LlamaGenDecoder(args.ckpt_path, device, args)

    # Setup data:
    crop_size = int(args.image_size * args.crop_range)
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ]
    )
    print("Constructing dataset...")
    dataset = build_dataset(args, transform=transform)
    if not args.single:
        sampler = DistributedSampler(
            dataset,
            num_replicas=dist.get_world_size(),
            rank=rank,
            shuffle=False,
            seed=args.global_seed,
        )
    else:
        sampler = None
    loader = DataLoader(
        dataset,
        batch_size=args.bsz,  # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize progress bar for rank 0 or single process
    progress_bar = tqdm(
        loader, desc="Processing", #disable=(not args.single and rank != 0)
    )

    total = 0
    x_concat = []
    y_concat = []
    index = 0
    for p, x, y in progress_bar:
        bsz = x.shape[0]
        if index > last_index:
            n_aug = x.shape[1]
            x = x.to(device).flatten(0, 1) # shape: (bsz * n_aug, 3, H, W)
            encoded_tokens = vq_model.encode(x).view(bsz, n_aug, -1).cpu().numpy()
            for i in range(bsz):
                x = encoded_tokens[i]
                x_concat.append(x)
                y_concat.append(
                    {"label": y[i].item() if isinstance(y, torch.Tensor) else y[i], "path": p[i]}
                )

        if total >= 500:
            if index > last_index:
                np.save(
                    f"{args.code_path}/{args.dataset}{args.image_size}_codes/{rank}_{index}.npy",
                    np.array(x_concat),
                )
                with open(
                    f"{args.code_path}/{args.dataset}{args.image_size}_labels/{rank}_{index}.json",
                    "w",
                ) as f:
                    json.dump(y_concat, f, indent=4)
                x_concat = []
                y_concat = []
            index += 1

        total += bsz

    if len(x_concat) > 0:
        np.save(
            f"{args.code_path}/{args.dataset}{args.image_size}_codes/{rank}_{index}.npy",
            np.array(x_concat),
        )
        with open(
            f"{args.code_path}/{args.dataset}{args.image_size}_labels/{rank}_{index}.json",
            "w",
        ) as f:
            json.dump(y_concat, f, indent=4)

    if not args.single:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="/root/projects/continuous_tokenizer/ImageNet/train",
    )
    parser.add_argument(
        "--vq-model", type=str, default="VQ-16"
    )
    parser.add_argument(
        "--codebook-size", type=int, default=16384
    )
    parser.add_argument(
        "--codebook-embed-dim", type=int, default=8
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="/root/projects/continuous_tokenizer/LlamaGen/vq_ds16_c2i.pt",
    )
    parser.add_argument(
        "--code-path",
        type=str,
        default="dataset/ImageNet-1k/llamagen_codes_384/",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument(
        "--image-size", type=int, choices=[256, 384, 448, 512], default=384
    )
    parser.add_argument(
        "--crop-range", type=float, default=1.1, help="expanding range of center crop"
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--bsz", type=int, default=1)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    args.code_path = convert_path(args.code_path, force=True)
    args.ckpt_path = convert_path(args.ckpt_path, force=True)
    assert os.path.exists(args.data_path), f"data path {args.data_path} does not exist!"
    main(args)
