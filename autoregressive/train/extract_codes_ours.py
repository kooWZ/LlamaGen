import os
from PIL import ImageFile

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

ours_path = os.path.abspath(
    os.path.join(
        llamagen_path,
        "external_tokenizers/postTok",
    )
)
sys.path.append(ours_path)
from modelling.tokenizer import VQ_models
from train.train_tokenizer import build_parser
from utils.misc import load_model_state_dict
from ruamel import yaml

from tqdm import tqdm
from glob import glob


def parse_args():
    parser = build_parser()
    parser.add_argument(
        "--weights",
        type=str,
        default=os.path.join(
            llamagen_path,
            "outputs/ckpts/postTok_sim128/0250000.pth",
        ),
    )
    args = parser.parse_args()
    args.config = os.path.join(
        llamagen_path,
        "autoregressive/train/configs/postTok/simvq128.yaml",
    )
    with open(args.config, "r", encoding="utf-8") as f:
        file_yaml = yaml.YAML()
        config_args = file_yaml.load(f)
        parser.set_defaults(**(config_args or {}))
    args = parser.parse_args([])
    args.encoder_local_ckpt = os.path.join(
        llamagen_path,
        "outputs/ckpts/dinov2/dinov2_vitb14_pretrain.pth",
    )
    args.decoder_local_ckpt = os.path.join(
        llamagen_path,
        "outputs/ckpts/dinov2/dinov2_vitl14_pretrain.pth",
    )
    args.repa_local_ckpt = os.path.join(
        llamagen_path,
        "outputs/ckpts/dinov2/dinov2_vitl14_pretrain.pth",
    )
    return args


def build_model(args, device):
    ModelClass = VQ_models[args.vq_model]
    model = ModelClass(
        image_size=args.image_size,
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim,
        codebook_l2_norm=args.codebook_l2_norm,
        commit_loss_beta=args.commit_loss_beta,
        entropy_loss_ratio=args.entropy_loss_ratio,
        vq_loss_ratio=args.vq_loss_ratio,
        kl_loss_weight=args.kl_loss_weight,
        dropout_p=args.dropout_p,
        enc_type=args.enc_type,
        encoder_model=args.encoder_model,
        dec_type=args.dec_type,
        decoder_model=args.decoder_model,
        num_latent_tokens=args.num_latent_tokens,
        enc_tuning_method=args.encoder_tuning_method,
        dec_tuning_method=args.decoder_tuning_method,
        enc_pretrained=args.encoder_pretrained,
        dec_pretrained=args.decoder_pretrained,
        enc_patch_size=args.encoder_patch_size,
        dec_patch_size=args.decoder_patch_size,
        tau=args.tau,
        repa=args.repa,
        repa_model=args.repa_model,
        repa_patch_size=args.repa_patch_size,
        repa_proj_dim=args.repa_proj_dim,
        repa_loss_weight=args.repa_loss_weight,
        repa_align=args.repa_align,
        num_codebooks=args.num_codebooks,
        enc_token_drop=args.enc_token_drop,
        enc_token_drop_max=args.enc_token_drop_max,
        cls_recon=args.cls_recon,
        cls_recon_weight=args.cls_recon_weight,
        aux_dec_model=args.aux_decoder_model,
        aux_loss_mask=args.aux_loss_mask,
        aux_hog_dec=args.aux_hog_decoder,
        aux_dino_dec=args.aux_dino_decoder,
        aux_clip_dec=args.aux_clip_decoder,
        aux_supcls_dec=args.aux_supcls_decoder,
        to_pixel=args.to_pixel,
        repa_local_ckpt=args.repa_local_ckpt,
        enc_local_ckpt=args.encoder_local_ckpt,
        dec_local_ckpt=args.decoder_local_ckpt,
    ).to(device)
    model.eval()

    weights = torch.load(args.weights, map_location="cpu")
    state_dict = weights.get("ema", weights.get("model", weights))
    state_dict = load_model_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=True)
    del weights
    return model


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
    dist.barrier()
    done_files = glob(
        os.path.join(
            args.code_path, f"{args.dataset}{args.image_size}_codes", f"{rank}_*.npy"
        )
    )
    last_index = len(done_files) - 1
    print(f"Rank {rank} has done {len(done_files)} files, last index {last_index}.")

    # create and load model
    model = build_model(args, device)
    model.requires_grad_(False)

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
        loader,
        desc="Processing",  # disable=(not args.single and rank != 0)
    )

    total = 0
    x_concat = []
    y_concat = []
    index = 0
    for p, x, y in progress_bar:
        if index > last_index:
            # x is [1, 10, 3, 256, 256]
            x = x.to(device).flatten(0, 1) # after flatten is [10, 3, 256, 256]
            _, _, info = model.encode(x)
            encoded_tokens = info[2]
            codes = encoded_tokens.flatten(0, 1) # [10, len]

            x = codes.cpu().numpy()
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
        default="dataset/ImageNet-1k/ours_0250000_codes/",
    )
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument(
        "--image-size", type=int, choices=[256, 384, 448, 512], default=256
    )
    parser.add_argument(
        "--crop-range", type=float, default=1.1, help="expanding range of center crop"
    )
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--single", action="store_true")
    args = parser.parse_args()

    args.code_path = convert_path(args.code_path, force=True)
    assert os.path.exists(args.data_path), f"data path {args.data_path} does not exist!"
    main(args)
