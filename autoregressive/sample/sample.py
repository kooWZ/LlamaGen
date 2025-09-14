import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
from torchvision.utils import save_image


import time
import argparse
import sys

sys.path.append("/root/kongly/AR/LlamaGen")
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate

sys.path.append("/root/kongly/AR/LlamaGen/external_tokenizers/flextok")
from external_tokenizers.flextok.flextok.flextok_wrapper import FlexTokFromHub
from external_tokenizers.flextok.flextok.utils.misc import get_bf16_context
from tqdm import tqdm
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("--gpt-model", type=str, default="GPT-Mini")
parser.add_argument(
    "--gpt-ckpt",
    type=str,
    default="/root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/30_0387841.pt",
)
parser.add_argument(
    "--gpt-type",
    type=str,
    choices=["c2i", "t2i"],
    default="c2i",
    help="class-conditional or text-conditional",
)
parser.add_argument("--from-fsdp", action="store_true")
parser.add_argument(
    "--cls-token-num", type=int, default=1, help="max token number of condition input"
)
parser.add_argument(
    "--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"]
)
parser.add_argument("--compile", action="store_true", default=True)
parser.add_argument(
    "--vq-ckpt",
    type=str,
    default="/root/kongly/ckpts/flextok_d12_d12_in1k/",
    help="ckpt path for vq model",
)
parser.add_argument(
    "--codebook-size",
    type=int,
    default=64000,
    help="codebook size for vector quantization",
)
parser.add_argument(
    "--codebook-embed-dim",
    type=int,
    default=8,
    help="codebook dimension for vector quantization",
)
parser.add_argument("--latent-size", type=int, default=256)
parser.add_argument("--num-classes", type=int, default=1000)
parser.add_argument("--cfg-scale", type=float, default=4.0)
parser.add_argument("--cfg-interval", type=float, default=-1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--top-k", type=int, default=2000, help="top-k value to sample with"
)
parser.add_argument(
    "--temperature", type=float, default=1.0, help="temperature value to sample with"
)
parser.add_argument(
    "--top-p", type=float, default=1.0, help="top-p value to sample with"
)
args = parser.parse_args([])


torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_grad_enabled(False)
device = "cuda"


class Args:
    def __init__(self, config):
        for key, value in config.items():
            setattr(self, key, value)


torch.serialization.add_safe_globals([Args])

def load_gpt(ckpt):
    precision = {"none": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        args.precision
    ]
    gpt_model = GPT_models[args.gpt_model](
        vocab_size=args.codebook_size,
        block_size=args.latent_size,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
    ).to(device=device, dtype=precision)

    checkpoint = torch.load(args.gpt_ckpt, map_location="cpu", weights_only=True)
    if args.from_fsdp:  # fspd
        model_weight = checkpoint
    elif "model" in checkpoint:  # ddp
        model_weight = checkpoint["model"]
    elif "module" in checkpoint:  # deepspeed
        model_weight = checkpoint["module"]
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
    else:
        raise Exception("please check model weight, maybe add --from-fsdp to run command")
    # if 'freqs_cis' in model_weight:
    #     model_weight.pop('freqs_cis')
    gpt_model.load_state_dict(model_weight, strict=False)
    gpt_model.eval()
    del checkpoint
    print(f"gpt model is loaded")

    if args.compile:
        print(f"compiling the model...")
        gpt_model = torch.compile(
            gpt_model, mode="reduce-overhead", fullgraph=True
        )  # requires PyTorch 2.0 (optional)
        print(f"compiling done")
    else:
        print(f"no need to compile model in demo")
    return gpt_model

gpt_30 = load_gpt(
    "/root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/30_0387841.pt"
)
gpt_60 = load_gpt(
    "/root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/60_0763171.pt"
)
gpt_92 = load_gpt(
    "/root/kongly/AR/LlamaGen/outputs/test/023-GPT-Mini/checkpoints/92_1163523.pt"
)
print("loading decoder")
vq_model = FlexTokFromHub.from_pretrained(args.vq_ckpt)
vq_model.to(device)
vq_model.eval()


from flextok.flextok_wrapper import FlexTokFromHub
from flextok.utils.demo import imgs_from_urls, denormalize, batch_to_pil
from flextok.utils.misc import detect_bf16_support, get_bf16_context, get_generator


def gen(gpt_model, name):
    class_labels = [207, 360, 387, 974, 88, 979, 417, 279]
    c_indices = torch.tensor(class_labels, device=device)

    print("sampling")
    t1 = time.time()
    index_sample = generate(
        gpt_model,
        c_indices,
        args.latent_size,
        cfg_scale=args.cfg_scale,
        cfg_interval=args.cfg_interval,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        sample_logits=True,
    )
    sampling_time = time.time() - t1
    print(f"gpt sampling takes about {sampling_time:.2f} seconds.")

    print("Reconstructions using all 256 tokens:")
    with get_bf16_context(True):
        reconst = vq_model.detokenize(
            [index_sample[t].unsqueeze(0) for t in range(index_sample.shape[0])],
            timesteps=25,  # Number of denoising steps
            guidance_scale=15,  # Classifier-free guidance scale
            perform_norm_guidance=True,  # APG, see https://arxiv.org/abs/2410.02416
            # Optionally control initial noise. Note that while the initial noise is deterministic, the rest of the model isn't.
            generator=get_generator(seed=0, device=device),
            verbose=False,  # Enable to show denoising progress bar with tqdm
        )
    batch_to_pil(reconst).save(f"{name}_256.png")
    print("Reconstructions using first 32 tokens:")
    with get_bf16_context(True):
        reconst = vq_model.detokenize(
            [index_sample[t][:32].unsqueeze(0) for t in range(index_sample.shape[0])],
            timesteps=25,  # Number of denoising steps
            guidance_scale=15,  # Classifier-free guidance scale
            perform_norm_guidance=True,  # APG, see https://arxiv.org/abs/2410.02416
            # Optionally control initial noise. Note that while the initial noise is deterministic, the rest of the model isn't.
            generator=get_generator(seed=0, device=device),
            verbose=False,  # Enable to show denoising progress bar with tqdm
        )
    batch_to_pil(reconst).save(f"{name}_32.png")

gen(gpt_30, 30)
gen(gpt_60, 60)
gen(gpt_92, 92)
