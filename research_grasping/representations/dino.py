"""Opt-in offline DINOv2/DINOv3 inference using local Hugging Face safetensors.

No download, remote code, pickle, image transform, or license acceptance is performed.
The caller supplies the preprocessed BCHW tensor and records its transform.
"""

import hashlib
from pathlib import Path

import numpy as np


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_local_dino(directory, expected_sha256, config_sha256, family, device="cpu"):
    """Load a single-file safetensors checkpoint and local config, strictly offline."""
    directory = Path(directory)
    weight = directory / "model.safetensors"
    if family not in ("dinov2", "dinov3"):
        raise ValueError("family must be dinov2 or dinov3")
    if not weight.is_file() or not (directory / "config.json").is_file():
        raise FileNotFoundError("local config.json and model.safetensors required")
    if len(expected_sha256) != 64 or sha256_file(weight) != expected_sha256.lower():
        raise ValueError("checkpoint SHA-256 mismatch")
    if (
        len(config_sha256) != 64
        or sha256_file(directory / "config.json") != config_sha256.lower()
    ):
        raise ValueError("config SHA-256 mismatch")
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise ImportError(
            "install compatible torch, transformers and safetensors separately"
        ) from exc
    config = AutoConfig.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False
    )
    if config.model_type != {"dinov2": "dinov2", "dinov3": "dinov3_vit"}[family]:
        raise ValueError("model family does not match configuration")
    model, info = AutoModel.from_pretrained(
        str(directory),
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(
        info.get(k)
        for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    ):
        raise ValueError("checkpoint does not exactly match the model configuration")
    model.to(device).eval()
    return model


def extract_tokens(model, pixel_values):
    """Extract CLS and spatial patches; input must already be preprocessed BCHW.

    Returns NumPy arrays. Patch count validation rejects unexpected token layouts.
    """
    import torch

    if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
        raise ValueError("expected BCHW RGB tensor")
    patch = model.config.patch_size
    if not isinstance(patch, int):
        raise TypeError("only square patch configurations are supported")
    h, w = pixel_values.shape[-2:]
    if h % patch or w % patch:
        raise ValueError(
            "input must be divisible by patch size; padding is not supported"
        )
    with torch.inference_mode():
        hidden = model(pixel_values=pixel_values).last_hidden_state
    count = (h // patch) * (w // patch)
    registers = int(getattr(model.config, "num_register_tokens", 0))
    if hidden.ndim != 3 or hidden.shape[1] != 1 + registers + count:
        raise ValueError("unexpected token layout")
    cls = hidden[:, 0].detach().cpu().numpy()
    patches = (
        hidden[:, 1 + registers :]
        .detach()
        .cpu()
        .numpy()
        .reshape(hidden.shape[0], h // patch, w // patch, -1)
    )
    return cls, patches


def masked_pool(patches, mask):
    p = np.asarray(patches)
    m = np.asarray(mask, dtype=bool)
    if p.ndim != 3 or m.shape != p.shape[:2] or not m.any():
        raise ValueError("mask must match patch grid and contain a patch")
    return p[m].mean(axis=0)
