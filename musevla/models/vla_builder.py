import os
import copy
import json
import torch
import transformers
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import hf_hub_download

import musevla.models.vla as vla
from musevla.utils.data_utils import read_dataset_statistics, GaussianNormalizer


def build_vla(configs, checkpoint_path=None):
    model_fn = getattr(vla, configs["vla_name"])
    model = model_fn(
        configs=configs,
        train_setup_configs=configs["train_setup"],
        act_model_configs=configs["action_model"],
        fwd_pred_next_n=configs["fwd_pred_next_n"],
        repeated_diffusion_steps=configs.get("repeated_diffusion_steps", 8),
        use_state=configs["use_state"],
        use_fov=configs.get("use_fov", True),
        checkpoint_path=checkpoint_path,
    )
    return model

def load_model(configs):
    model_load_path = configs.get("model_load_path", None)
    checkpoint_path = None
    
    if model_load_path is not None:
        # Check if model_load_path is a file or directory
        if os.path.isfile(model_load_path):
            # If it's a file, use its directory for tokenizer
            checkpoint_path = model_load_path
        else:
            # If it's a directory, look for weights.pt inside
            checkpoint_path = os.path.join(model_load_path, "weights.pt")
    
    model = build_vla(
        configs=configs,
        checkpoint_path=checkpoint_path,
    )
    
    if checkpoint_path is not None:
        model = load_vla_checkpoint(model, checkpoint_path)

    return model

def load_vla_checkpoint(model, checkpoint_path, map_location="cpu"):
    # print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    # Use strict=False to handle architecture differences between checkpoint and current model
    # (e.g., checkpoint has fov_encoder/cognition_token but current config doesn't use them)
    with torch.no_grad():
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)
    
    if missing_keys:
        print(f"Warning: Missing keys in checkpoint (will be randomly initialized): {missing_keys}")
    if unexpected_keys:
        print(f"Warning: Unexpected keys in checkpoint (will be ignored): {unexpected_keys}")
    
    # print("Checkpoint loaded successfully")
    return model

