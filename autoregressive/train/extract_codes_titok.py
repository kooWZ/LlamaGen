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

titok_path = os.path.abspath(
    os.path.join(llamagen_path, "external_tokenizers/TiTok")
)
sys.path.append(titok_path)
from modeling.titok import TiTok

from tqdm import tqdm
from glob import glob


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

    done_files = glob(
        os.path.join(
            args.code_path, f"{args.dataset}{args.image_size}_codes", f"{rank}_*.npy"
        )
    )
    last_index = len(done_files) - 1
    print(f"Rank {rank} has done {len(done_files)} files, last index {last_index}.")

    # create and load model
    titok_tokenizer = TiTok.from_pretrained("yucornetto/tokenizer_titok_l32_imagenet")
    titok_tokenizer.eval()
    titok_tokenizer.requires_grad_(False)

    # Setup data:
    crop_size = int(args.image_size * args.crop_range)
    single_random_crop = transforms.RandomResizedCrop(
        size=args.image_size,  # 输出尺寸
    )
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(
                lambda pil_image: torch.stack(
                    [
                        transforms.ToTensor()(single_random_crop(pil_image))
                        for _ in range(10)  # 生成 10 个随机裁剪
                    ]
                )
            ),
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
        batch_size=1,  # important!
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Initialize progress bar for rank 0 or single process
    progress_bar = tqdm(
        loader, desc="Processing", disable=(not args.single and rank != 0)
    )

    total = 0
    x_concat = []
    y_concat = []
    index = 0
    for p, x, y in progress_bar:
        if index > last_index:
            x = x.to(device)
            x_all = x.flatten(0, 1)
            encoded_tokens = titok_tokenizer.encode(x_all.to(device))[1][
                "min_encoding_indices"
            ]
            codes = torch.cat(encoded_tokens, dim=0)

            x = codes.detach().cpu().numpy()
            x_concat.append(x)
            y_concat.append(
                {"label": y.item() if isinstance(y, torch.Tensor) else y, "path": p}
            )

        if total % 500 == 0:
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

        total += 1

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
        default="/root/kongly/AR/LlamaGen/dataset/ImageNet-1k/data/train",
    )
    parser.add_argument(
        "--code-path",
        type=str,
        default="/root/kongly/AR/LlamaGen/dataset/ImageNet-1k/flextok_codes/large/random_crop",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument(
        "--image-size", type=int, choices=[256, 384, 448, 512], default=256
    )
    parser.add_argument(
        "--crop-range", type=float, default=1.1, help="expanding range of center crop"
    )
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()
    main(args)
