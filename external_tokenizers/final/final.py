from pathlib import Path
from omegaconf import OmegaConf
import torch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../../")))
from modelling.tokenizer import ModelArgs, VFMTokModel
import numpy as np
print(f"Using repo root: {Path.cwd()}")

def load_model_args_from_yaml(config_path: str) -> ModelArgs:
    """Load YAML,过滤出 ModelArgs 需要的字段并构造 dataclass。"""
    cfg = OmegaConf.load(config_path)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    valid_fields = ModelArgs.__dataclass_fields__.keys()
    args_dict = {k: cfg_dict[k] for k in valid_fields if k in cfg_dict}
    if "transformer_config" not in args_dict or not args_dict["transformer_config"]:
        raise ValueError("ModelArgs requires 'transformer_config'.")
    return ModelArgs(**args_dict)


def build_vfmtok_model(config_path: str) -> VFMTokModel:
    """Convenience wrapper that returns (model_args, model)。"""
    model_args = load_model_args_from_yaml(config_path)
    model = VFMTokModel(model_args)
    return model_args, model

CONFIG_PATH = "/root/projects/continuous_tokenizer/configs/vfmtok-semtok.yaml"

def build_and_load_model(config_path: str, ckpt_path: str) -> VFMTokModel:
    model_args, vq_model = build_vfmtok_model(config_path)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "ema" in checkpoint:
        model_weight = checkpoint["ema"]
        print("Using 'ema' weights from checkpoint.")
    elif "model" in checkpoint:
        model_weight = checkpoint["model"]
        print("Using 'model' weights from checkpoint.")
    elif "state_dict" in checkpoint:
        model_weight = checkpoint["state_dict"]
        print("Using 'state_dict' weights from checkpoint.")
    else:
        raise RuntimeError(
            "Unknown checkpoint format, no 'ema' / 'model' / 'state_dict' keys found."
        )
    model_true_weight = {}
    for k,v in model_weight.items():
        if not (k.startswith("decoder") or k.startswith("quantize")):
            model_true_weight[k] = v

    missings, unexpected = vq_model.load_state_dict(model_true_weight, strict=False)
    if missings:
        print("[Warning] Missing keys in state_dict:")
        for k in missings:
            print("   ", k)
    if unexpected:
        print("[Warning] Unexpected keys in state_dict:")
        for k in unexpected:
            print("   ", k)
    return model_args, vq_model