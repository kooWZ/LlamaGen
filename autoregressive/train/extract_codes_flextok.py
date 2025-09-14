# Modified from:
#   fast-DiT: https://github.com/chuanyangjin/fast-DiT/blob/main/extract_features.py
import os

os.environ["TORCH_COMPILE_DISABLE"] = "0"

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import json
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_grad_enabled(False)

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
import argparse
import sys

sys.path.append("/root/kongly/AR/LlamaGen")
from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from dataset.build import build_dataset

sys.path.append("/root/kongly/AR/LlamaGen/external_tokenizers/flextok")
from external_tokenizers.flextok.flextok.flextok_wrapper import FlexTokFromHub
from external_tokenizers.flextok.flextok.utils.misc import get_bf16_context
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

    done_files = glob(os.path.join(args.code_path, f"{args.dataset}{args.image_size}_codes", f"{rank}_*.npy"))
    last_index = len(done_files) - 1
    print(f"Rank {rank} has done {len(done_files)} files, last index {last_index}.")

    # create and load model
    vq_model = FlexTokFromHub.from_pretrained(
        "/root/kongly/ckpts/flextok_d12_d12_in1k/"
    )
    vq_model.to(device)
    vq_model.eval()

    # Setup data:
    crop_size = int(args.image_size * args.crop_range)
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
    #     transforms.TenCrop(args.image_size), # this is a tuple of PIL Images
    #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), # returns a 4D tensor
    # ])
    single_random_crop = transforms.RandomResizedCrop(
        size=args.image_size,        # 输出尺寸
    )
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, crop_size)),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(
                lambda pil_image: torch.stack(
                    [
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(
                            transforms.ToTensor()(single_random_crop(pil_image))
                        )
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
            with get_bf16_context(True):
                tokens_list = vq_model.tokenize(x_all)

            codes = torch.cat(tokens_list, dim=0)

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
                with open(f"{args.code_path}/{args.dataset}{args.image_size}_labels/{rank}_{index}.json", 'w') as f:
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
        with open(f"{args.code_path}/{args.dataset}{args.image_size}_labels/{rank}_{index}.json", 'w') as f:
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
        default="/root/kongly/AR/LlamaGen/dataset/ImageNet-1k/flextok_codes/random_crop",
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
