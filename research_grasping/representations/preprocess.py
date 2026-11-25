"""Explicit versioned RGB preprocessing; caller must select protocol parameters."""

import numpy as np


def preprocess_rgb(rgb, resized_size, crop_size, mean, std):
    """Resize bilinear (align_corners=False), centre crop and normalize RGB uint8.

    Return BCHW CPU tensor, pixel-centre affine and cache-ready specification.
    No antialiasing, padding, colour conversion or automatic model defaults.
    """
    import torch
    import torch.nn.functional as F

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be HWC uint8")
    h, w = rgb.shape[:2]
    rh, rw = resized_size
    ch, cw = crop_size
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    if (
        min(h, w, rh, rw, ch, cw) <= 0
        or ch > rh
        or cw > rw
        or mean.shape != (3,)
        or std.shape != (3,)
        or not np.isfinite(mean).all()
        or not np.isfinite(std).all()
        or np.any(std <= 0)
    ):
        raise ValueError("invalid preprocessing parameters")
    top, left = (rh - ch) // 2, (rw - cw) // 2
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).float() / 255
    x = F.interpolate(
        x, size=(rh, rw), mode="bilinear", align_corners=False, antialias=False
    )
    x = x[:, :, top : top + ch, left : left + cw]
    x = (x - torch.tensor(mean, dtype=x.dtype)[None, :, None, None]) / torch.tensor(
        std, dtype=x.dtype
    )[None, :, None, None]
    sx, sy = rw / w, rh / h
    affine = np.array(
        [[sx, 0, (sx - 1) / 2 - left], [0, sy, (sy - 1) / 2 - top], [0, 0, 1.0]]
    )
    spec = {
        "version": "rgb-bilinear-center-v1",
        "torch": torch.__version__,
        "input_size": [h, w],
        "resized_size": [rh, rw],
        "crop_size": [ch, cw],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "antialias": False,
    }
    return x, affine, spec
