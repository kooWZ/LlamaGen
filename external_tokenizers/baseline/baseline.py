from pathlib import Path
from omegaconf import OmegaConf
import torch
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../../../")))
from vfmtok.tokenizer.vq_model import OldModelArgs, OldVQModel,VQ_models
import numpy as np
print(f"Using repo root: {Path.cwd()}")



def build_and_load_model(config_path: str, ckpt_path: str):
    transformer_config = OmegaConf.load(config_path)
    vq_model = VQ_models['VQ-16'](
        codebook_size=16384,
        z_channels=512,
        codebook_embed_dim=12,
        transformer_config = transformer_config)
    vq_model.eval()
    vq_model.freeze()
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    del checkpoint["ema"]['slot_quantize.codebook_used']
    m1, u1 = vq_model.load_state_dict(checkpoint["ema"], strict=False)
    del checkpoint
    return None, vq_model