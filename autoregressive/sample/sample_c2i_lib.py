import argparse
import torch
import torch.distributed as dist
from tqdm import tqdm
import os
import time
import numpy as np
import math
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
llamagen_path = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(llamagen_path)
from autoregressive.models.generate import generate
from autoregressive.models.gpt import GPT_models


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class FinalDecoder:
    def __init__(self, config_path, ckpt_path, device):
        final_path = os.path.abspath(os.path.join(llamagen_path, "external_tokenizers/final/"))
        sys.path.append(final_path)
        from final import build_and_load_model
        self.model = build_and_load_model(config_path, ckpt_path)[1].to(device)

    def encode(self, img):
        return self.model.encode_to_ids(img) # tensor of [1, 576]

    def decode(self, ids, args):
        return self.model.decode_from_ids(ids).detach() # tensor of [1, 3, 256, 256]

    def denormalize(self, recon): # recon should be [3, 256, 256]
        dfi = torch.clamp((recon + 1) / 2, 0, 1)
        dfiimg = (dfi.permute(1, 2, 0) * 255).to(torch.uint8).cpu()
        return dfiimg

class FlexTokDecoder:
    def __init__(self, vq_ckpt, device):
        flextok_path = os.path.abspath(os.path.join(llamagen_path, "external_tokenizers/flextok"))
        sys.path.append(flextok_path)
        from flextok.flextok_wrapper import FlexTokFromHub

        self.device = device
        self.vq_model = FlexTokFromHub.from_pretrained(vq_ckpt)
        self.vq_model.to(self.device)
        self.vq_model.eval()

    def decode(self, arr, args):
        from flextok.utils.misc import get_bf16_context, get_generator

        with get_bf16_context(True):
            reconst = self.vq_model.detokenize(
                [arr[t].unsqueeze(0) for t in range(arr.shape[0])],
                timesteps=args.decoder_timesteps,  # Number of denoising steps
                guidance_scale=args.decoder_guidance_scale,  # Classifier-free guidance scale
                perform_norm_guidance=True,  # APG, see https://arxiv.org/abs/2410.02416
                # Optionally control initial noise. Note that while the initial noise is deterministic, the rest of the model isn't.
                generator=get_generator(seed=0, device=self.device),
                verbose=False,  # Enable to show denoising progress bar with tqdm
            )
        return reconst

    def denormalize(self, recon):
        from flextok.utils.demo import denormalize

        img_tensor = denormalize(recon).clamp(0, 1)
        img_converted = (img_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu()
        return img_converted


class TiTokDecoder:
    def __init__(self, vq_ckpt, device):
        titok_path = os.path.abspath(os.path.join(llamagen_path, "external_tokenizers/TiTok"))
        sys.path.append(titok_path)
        from modeling.titok import TiTok

        self.device = device
        self.titok_tokenizer = TiTok.from_pretrained(vq_ckpt)
        self.titok_tokenizer = self.titok_tokenizer.to(self.device)
        self.titok_tokenizer.eval()
        self.titok_tokenizer.requires_grad_(False)

    def decode(self, arr, args):
        reconst = self.titok_tokenizer.decode_tokens(arr.to(self.device))
        return reconst

    def denormalize(self, recon):
        img_tensor = recon.clamp(0, 1)
        img_converted = (img_tensor.permute(1, 2, 0) * 255).to(torch.uint8).cpu()
        return img_converted


class Args:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)

    def __getattr__(self, name):
        print(f"Warning: {name} not found in Args, returning None")
        return None


class OurDecoder:
    def build_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, default='configs/tokenizer/cnn_llamagen_vq16.yaml', help="config file used to specify parameters")

        parser.add_argument("--exp-index", type=str, default=None, help="experiment index")
        parser.add_argument("--data-path", type=str, default="ImageNet2012/train")
        parser.add_argument("--eval-data-path", type=str, default="ImageNet2012/val")
        parser.add_argument("--cloud-save-path", type=str, help='please specify a cloud disk path, if not, local path')
        parser.add_argument("--no-local-save", type=str2bool, default=False, help='no save checkpoints to local path for limited disk volume')
        parser.add_argument("--vq-model", type=str, default="VQ-16")
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

    def __init__(self, config_file, vq_ckpt, device):
        from ruamel import yaml

        parser = self.build_parser()
        with open(config_file, "r", encoding="utf-8") as f:
            file_yaml = yaml.YAML()
            config_args = file_yaml.load(f)
            parser.set_defaults(**config_args)
        self.dec_args = parser.parse_args([])
        self.dec_args.encoder_local_ckpt = os.path.join(
            llamagen_path,
            self.dec_args.encoder_local_ckpt,
        )
        self.dec_args.decoder_local_ckpt = os.path.join(
            llamagen_path,
            self.dec_args.decoder_local_ckpt,
        )
        self.dec_args.repa_local_ckpt = os.path.join(
            llamagen_path,
            self.dec_args.repa_local_ckpt,
        )
        self.device = device
        self.vq_model = self._build_model(vq_ckpt)
        self.vq_model.eval()

    def _build_model(self, vq_ckpt):
        ours_path = os.path.abspath(
            os.path.join(llamagen_path, "external_tokenizers/postTok")
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
        from postTok.utils.misc import load_model_state_dict

        args = self.dec_args
        ModelClass = VQ_models[args.vq_model]
        model = ModelClass(
            image_size=args.image_size,
            codebook_size=args.codebook_size,
            codebook_embed_dim=args.codebook_embed_dim,
            codebook_l2_norm=args.codebook_l2_norm,
            commit_loss_beta=args.commit_loss_beta,
            entropy_loss_ratio=args.entropy_loss_ratio,
            vq_loss_ratio=args.vq_loss_ratio,
            kl_loss_weight=args.kl_loss_weight, #
            dropout_p=args.dropout_p, #
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
            tau=args.tau, # 
            repa=args.repa,
            repa_model=args.repa_model,
            repa_patch_size=args.repa_patch_size,
            repa_proj_dim=args.repa_proj_dim,
            repa_loss_weight=args.repa_loss_weight,
            repa_align=args.repa_align,
            num_codebooks=args.num_codebooks,
            enc_token_drop=args.enc_token_drop, # 0.0
            enc_token_drop_max=args.enc_token_drop_max, # 0.6
            cls_recon=args.cls_recon,
            cls_recon_weight=args.cls_recon_weight,
            aux_dec_model=args.aux_decoder_model, #
            aux_loss_mask=args.aux_loss_mask, #
            aux_hog_dec=args.aux_hog_decoder, # 
            aux_dino_dec=args.aux_dino_decoder, # 
            aux_clip_dec=args.aux_clip_decoder, # 
            aux_supcls_dec=args.aux_supcls_decoder, # 
            to_pixel=args.to_pixel, # 
            repa_local_ckpt=args.repa_local_ckpt,
            enc_local_ckpt=args.encoder_local_ckpt,
            dec_local_ckpt=args.decoder_local_ckpt,
        ).to(self.device)
        model.eval()

        weights = torch.load(vq_ckpt, map_location="cpu")
        state_dict = weights.get("ema", weights.get("model", weights))
        state_dict = load_model_state_dict(state_dict)
        model.load_state_dict(state_dict, strict=True)
        del weights
        return model

    def decode(self, arr, args):
        with torch.no_grad():
            reconst = self.vq_model.decode_from_ids(arr.to(self.device).contiguous()).cpu()
        return reconst

    def denormalize(self, recon):
        recon_image = torch.clamp((recon + 1) / 2, 0, 1)
        img_converted = (recon_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu()
        return img_converted

@torch.compiler.disable(recursive=True)
def do_sample(ckpt_path, args, rank, device, npz_path):
    if args.drop_path_rate > 0.0:
        dropout_p = 0.0
    else:
        dropout_p = args.dropout_p
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=args.latent_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=dropout_p,
        ffn_dropout_p=dropout_p,
        drop_path_rate=args.drop_path_rate,
        token_dropout_p=args.token_dropout_p,
        use_liger=args.use_liger,
        fp32_attention=False,
    ).to(device)
    if ckpt_path is not None:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if args.from_fsdp:  # fsdp
            model_weight = checkpoint
        elif "model" in checkpoint:  # ddp
            model_weight = checkpoint["model"]
        elif "module" in checkpoint:  # deepspeed
            model_weight = checkpoint["module"]
        elif "state_dict" in checkpoint:
            model_weight = checkpoint["state_dict"]
        else:
            raise Exception(
                "please check model weight, maybe add --from-fsdp to run command"
            )
        gpt_model.load_state_dict(model_weight, strict=False)
        del checkpoint
    gpt_model.eval()

    vq_ckpt = args.vq_ckpt
    if not os.path.exists(vq_ckpt):
        vq_ckpt = os.path.join(llamagen_path, vq_ckpt)
    assert os.path.exists(vq_ckpt), f"VQ model checkpoint {vq_ckpt} does not exist!"
    if args.decoder_type == "flextok":
        decoder = FlexTokDecoder(vq_ckpt, device)
    elif args.decoder_type == "titok":
        decoder = TiTokDecoder(vq_ckpt, device)
    elif args.decoder_type == "ours":
        vq_config = args.vq_config
        if not os.path.exists(vq_config):
            vq_config = os.path.join(llamagen_path, vq_config)
        assert os.path.exists(vq_config), f"VQ model config {vq_config} does not exist!"
        decoder = OurDecoder(vq_config, vq_ckpt, device)
    elif args.decoder_type == "final":
        vq_config = args.vq_config
        if not os.path.exists(vq_config):
            vq_config = os.path.join(llamagen_path, vq_config)
        assert os.path.exists(vq_config), f"VQ model config {vq_config} does not exist!"
        decoder = FinalDecoder(vq_config, vq_ckpt, device)
    else:
        raise NotImplementedError(f"Decoder type {args.decoder_type} not implemented")
    dist.barrier()

    n = args.eval_per_gpu_batch_size
    global_batch_size = n * dist.get_world_size()
    if rank == 0:
        print(f"Global batch size: {global_batch_size} ({n} per GPU on {dist.get_world_size()} GPUs)")
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_samples = int(
        math.ceil(args.eval_num_fid_samples / global_batch_size) * global_batch_size
    )
    if rank == 0:
        print(f"Total number of images that will be sampled: {total_samples}")
    assert (
        total_samples % dist.get_world_size() == 0
    ), "total_samples must be divisible by world_size"
    samples_needed_this_gpu = int(total_samples // dist.get_world_size())
    assert (
        samples_needed_this_gpu % n == 0
    ), "samples_needed_this_gpu must be divisible by the per-GPU batch size"
    iterations = int(samples_needed_this_gpu // n)
    pbar = range(iterations)
    pbar = tqdm(pbar, desc=f"Rank {rank}")# if rank == 0 else pbar
    total = 0

    # Use CPU storage and periodic gathering to manage memory efficiently
    batch_samples = []  # Temporary storage for current batch
    batch_labels = []  # Temporary storage for corresponding class indices

    # rank0 maintains all collected samples
    if rank == 0:
        final_samples_list = []
        final_labels_list = []
        print(
            f"Will gather samples every {args.eval_gather_freq} iterations to manage memory efficiently"
        )

    # Store image shape once we know it
    img_shape = None

    for iteration in pbar:
        # Sample inputs:
        c_indices = torch.randint(0, args.num_classes, (n,), device=device)

        index_sample = generate(
            gpt_model,
            c_indices,
            args.latent_size if args.eval_latent_size is None else args.eval_latent_size,
            cfg_scale=args.cfg_scale,
            cfg_interval=args.cfg_interval,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            sample_logits=True,
        )
        reconst = decoder.decode(index_sample, args)

        # Convert tensors directly to CPU numpy arrays to save GPU memory
        batch_data = []
        batch_class_indices = []
        for i in range(n):
            img_converted = decoder.denormalize(reconst[i])
            batch_data.append(img_converted)
            # Store corresponding class index
            batch_class_indices.append(c_indices[i].int().cpu())  # .numpy()

        # Store in CPU memory and record image shape
        if batch_data and img_shape is None:
            img_shape = batch_data[0].shape
        batch_samples.extend(batch_data) # tensorfy option
        batch_labels.extend(batch_class_indices)
        total += global_batch_size

        # Periodic gathering to rank0 to manage memory
        if (iteration + 1) % args.eval_gather_freq == 0 or (
            iteration + 1
        ) == iterations:
            start_gather_time = time.time()

            # Convert current batch to numpy arrays for gathering
            rank_batch_images = torch.stack(batch_samples).to(device)
            rank_batch_labels = torch.tensor(batch_labels).to(device)
            batch_samples = []  # Clear after gathering
            batch_labels = []  # Clear after gathering

            if rank == 0:
                print(f"Gathering samples from iteration {iteration + 1}...")

            # Gather this batch from all ranks (both images and labels)
            gathered_images = [
                torch.zeros_like(rank_batch_images, device=device)
                for _ in range(dist.get_world_size())
            ]
            gathered_labels = [
                torch.zeros_like(rank_batch_labels, device=device)
                for _ in range(dist.get_world_size())
            ]
            dist.barrier()
            dist.all_gather(
                gathered_images, rank_batch_images
            )
            dist.all_gather(
                gathered_labels, rank_batch_labels
            )
            # rank0 processes and stores the gathered data
            if rank == 0:
                for rank_images, rank_labels in zip(gathered_images, gathered_labels):
                    if rank_images.shape[0] > 0:
                        final_samples_list.append(rank_images.cpu())
                        final_labels_list.append(rank_labels.cpu())
                total_collected = sum(arr.shape[0] for arr in final_samples_list)
                print(
                    f"Rank0: Gather took {time.time() - start_gather_time} seconds. Collected {total_collected} total samples and labels so far"
                )

            # Synchronize all processes
            dist.barrier()

    # Final processing and NPZ creation (data already collected via periodic gathering)
    del decoder
    del gpt_model
    torch.cuda.empty_cache()
    dist.barrier()

    if rank == 0:
        if final_samples_list and final_labels_list:
            print("Concatenating all collected samples and labels...")
            final_samples = np.concatenate([arr.numpy() for arr in final_samples_list], axis=0)
            final_labels = np.concatenate([arr.numpy() for arr in final_labels_list], axis=0)

            # Truncate to exact number of samples needed (maintain correspondence)
            final_samples = final_samples[: args.eval_num_fid_samples]
            final_labels = final_labels[: args.eval_num_fid_samples]

            # Save directly to NPZ with both images and labels
            np.savez(npz_path, arr_0=final_samples, labels=final_labels)
            print(f"Saved .npz file to {npz_path}")
            print(f"  Images shape: {final_samples.shape}")
            print(f"  Labels shape: {final_labels.shape}")
            print("Memory efficient generation with labels completed!")
        else:
            print("!!!WARNING: No samples collected!")
