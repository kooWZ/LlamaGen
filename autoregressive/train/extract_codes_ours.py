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

external_path = os.path.abspath(
    os.path.join(
        llamagen_path,
        "external_tokenizers",
    )
)
sys.path.append(external_path)
from postTok.utils.misc import load_model_state_dict, str2bool
from ruamel import yaml

from tqdm import tqdm
from glob import glob

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")
    
    parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
    parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
    parser.add_argument("--eval-data-path", type=str, default="ImageNet2012/val")
    parser.add_argument("--cloud-save-path", type=str, help='please specify a cloud disk path, if not, local path')
    parser.add_argument("--no-local-save", type=str2bool, default=False, help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--finetune", type=str2bool, default=False, help="finetune a pre-trained vq model")
    parser.add_argument("--ema", type=str2bool, default=True, help="whether using ema training")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--codebook-l2-norm", type=str2bool, default=True, help="l2 norm codebook")
    parser.add_argument("--codebook-weight", type=float, default=1.0, help="codebook loss weight for vector quantization")
    parser.add_argument("--entropy-loss-ratio", type=float, default=0.0, help="entropy loss ratio in codebook loss")
    parser.add_argument("--vq-loss-ratio", type=float, default=1.0, help="vq loss ratio in codebook loss")
    parser.add_argument("--commit-loss-beta", type=float, default=0.25, help="commit loss beta in codebook loss")
    parser.add_argument("--reconstruction-weight", type=float, default=1.0, help="reconstruction loss weight of image pixel")
    parser.add_argument("--reconstruction-loss", type=str, default='l2', help="reconstruction loss type of image pixel")
    parser.add_argument("--kl-loss-weight", type=float, default=0.000001)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--num-codebooks", type=int, default=1)
    
    parser.add_argument("--perceptual-weight", type=float, default=1.0, help="perceptual loss weight of LPIPS")
    parser.add_argument("--perceptual-loss", type=str, default='vgg', help="perceptual loss type of LPIPS", choices=['vgg', 'timm', 'tv'])
    parser.add_argument("--perceptual-model", type=str, default='vgg', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-dino-variants", type=str, default='depth12_no_train', help="perceptual loss model of LPIPS")
    parser.add_argument("--perceptual-intermediate-loss", type=str2bool, default=False, help="perceptual loss compute at intermedia features of LPIPS")
    parser.add_argument("--perceptual-logit-loss", type=str2bool, default=False, help="perceptual loss compute at logits of LPIPS")
    parser.add_argument("--perceptual-resize", type=str2bool, default=False, help="perceptual loss compute at resized images of LPIPS")
    parser.add_argument("--perceptual-warmup", type=int, default=None, help="iteration to warmup perceptual loss")
    
    parser.add_argument("--disc-weight", type=float, default=0.5, help="discriminator loss weight for gan training")
    parser.add_argument("--disc-start", type=int, default=20000, help="iteration to start discriminator training and loss")
    parser.add_argument("--disc-dim", type=int, default=64, help="discriminator channel base dimension")
    parser.add_argument("--disc-type", type=str, choices=['patchgan', 'stylegan', 'maskbit', 'dino'], default='patchgan', help="discriminator type")
    parser.add_argument("--disc-loss", type=str, choices=['hinge', 'vanilla', 'non-saturating'], default='hinge', help="discriminator loss")
    parser.add_argument("--gen-loss", type=str, choices=['hinge', 'non-saturating'], default='hinge', help="generator loss for gan training")
    parser.add_argument("--lecam-loss-weight", type=float, default=None)
    parser.add_argument("--use-diff-aug",type=str2bool, default=False)
    parser.add_argument("--disc-cr-loss-weight", type=float, default=0.0, help="discriminator consistency loss weight for gan training")
    parser.add_argument("--disc-adaptive-weight",type=str2bool, default=False)
    
    parser.add_argument("--compile", type=str2bool, default=False)
    parser.add_argument("--dropout-p", type=float, default=0.0, help="dropout_p")
    parser.add_argument("--results-dir", type=str, default="results_tokenizer_image")
    parser.add_argument("--dataset", type=str, default='imagenet')
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_warmup_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler", type=str, default='none')
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--vis-every", type=int, default=5000)
    parser.add_argument("--ckpt-every", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='bf16', choices=["none", "fp16", "bf16"]) 

    parser.add_argument("--enc-type", type=str, default="cnn")
    parser.add_argument("--dec-type", type=str, default="cnn")
    parser.add_argument("--num-latent-tokens", type=int, default=None)
    parser.add_argument("--encoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='encoder model name')
    parser.add_argument("--decoder-model", type=str, default='vit_small_patch14_dinov2.lvd142m', help='decoder model name')
    parser.add_argument("--encoder-tuning-method", type=str, default='full', help='tuning method for encoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--decoder-tuning-method", type=str, default='full', help='tuning method for decoder', choices=['full', 'lora', 'frozen'])
    parser.add_argument("--encoder-pretrained", type=str2bool, default=True, help='load pre-trained weight for encoder')
    parser.add_argument("--decoder-pretrained", type=str2bool, default=False, help='load pre-trained weight for decoder')
    parser.add_argument("--encoder-local-ckpt", type=str, default="", help="Path to local encoder checkpoint (.pth) to avoid downloads.")
    parser.add_argument("--decoder-local-ckpt", type=str, default="", help="Path to local decoder checkpoint (.pth) to avoid downloads.")
    parser.add_argument("--encoder-patch-size", type=int, default=16, help='encoder patch size')
    parser.add_argument("--decoder-patch-size", type=int, default=16, help='decoder patch size')
    parser.add_argument("--to-pixel", type=str, default="linear")
    
    # repa
    parser.add_argument("--repa", type=str2bool, default=False, help='use repa')
    parser.add_argument('--repa-model', type=str, default='vit_base_patch16_224', help='repa model name')
    parser.add_argument('--repa-patch-size', type=int, default=16, help='repa patch size')
    parser.add_argument('--repa-proj-dim', type=int, default=1024, help='repa embed dim')
    parser.add_argument('--repa-loss-weight', type=float, default=0.1, help='repa loss weight')
    parser.add_argument('--repa-align', type=str, default='global', help='align repa feature', choices=['global', 'avg_1d', 'avg_2d', 'avg_1d_shuffle'])
    parser.add_argument('--repa-local-ckpt', type=str, default="", help="Path to local REPA teacher checkpoint (.pth).")
    parser.add_argument('--cls-recon', type=str2bool, default=False, help='enable CLS reconstruction with REPA teacher')
    parser.add_argument('--cls-recon-weight', type=float, default=0.03, help='CLS reconstruction loss weight')
    
    # aux decoder
    parser.add_argument("--aux-decoder-model", type=str, default='vit_tiny_patch14_dinov2_movq', help='aux decoder model name')
    parser.add_argument("--aux-loss-mask", type=str2bool, default='False', help='compute loss only at mask region')
    parser.add_argument("--aux-hog-decoder", type=str2bool, default=True, help='aux decoder hog decoder')
    parser.add_argument("--aux-dino-decoder", type=str2bool, default=True, help='aux decoder dino decoder')
    parser.add_argument("--aux-clip-decoder", type=str2bool, default=True, help='aux decoder hog decoder')
    parser.add_argument("--aux-supcls-decoder", type=str2bool, default=True, help='aux decoder hog decoder')
    
    # mask modeling
    # make sure drop is 0.0 for not using mask modeling
    parser.add_argument("--enc-token-drop", type=float, default=0.0, help='encoder token drop')
    parser.add_argument("--enc-token-drop-max", type=float, default=0.75, help='maximum drop rate')
    
    return parser


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
    config_file = os.path.join(
        llamagen_path,
        "autoregressive/train/configs/postTok/simvq128.yaml",
    )
    with open(config_file, "r", encoding="utf-8") as f:
        file_yaml = yaml.YAML()
        config_args = file_yaml.load(f)
        parser.set_defaults(**config_args)
    args = parser.parse_args([])
    args.config = config_file
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


def build_model(device):
    args = parse_args()
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
    model = build_model(device)
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
