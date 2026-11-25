"""Offline architecture smoke tests: random weights, never scientific evidence."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
t = pytest.importorskip("transformers")
pytest.importorskip("safetensors")
from safetensors.torch import load_file, save_file

from research_grasping.representations.dino import (
    extract_tokens,
    load_local_dino,
    sha256_file,
)
from research_grasping.representations.preprocess import preprocess_rgb


@pytest.mark.parametrize("family", ["dinov2", "dinov3"])
def test_local_roundtrip(family, tmp_path):
    kwargs = {
        "image_size": 32,
        "patch_size": 8,
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 64,
    }
    if family == "dinov2":
        model = t.Dinov2Model(t.Dinov2Config(**kwargs))
    else:
        model = t.DINOv3ViTModel(t.DINOv3ViTConfig(**kwargs, num_register_tokens=2))
    model.save_pretrained(tmp_path, safe_serialization=True)
    weight = tmp_path / "model.safetensors"
    config = tmp_path / "config.json"
    wh, ch = sha256_file(weight), sha256_file(config)
    with pytest.raises(ValueError):
        load_local_dino(tmp_path, "0" * 64, ch, family)
    with pytest.raises(ValueError):
        load_local_dino(tmp_path, wh, "0" * 64, family)
    loaded = load_local_dino(tmp_path, wh, ch, family)
    x, affine, spec = preprocess_rgb(
        np.zeros((40, 48, 3), np.uint8), (40, 40), (32, 32), [0, 0, 0], [1, 1, 1]
    )
    assert spec["version"] == "rgb-bilinear-center-v1" and x.shape == (1, 3, 32, 32)
    np.testing.assert_allclose(affine @ [0, 0, 1], [40 / 48 / 2 - 0.5 - 4, -4, 1])
    cls, patches = extract_tokens(loaded, x)
    assert cls.shape == (1, 32) and patches.shape == (1, 4, 4, 32)
    tensors = load_file(weight)
    tensors.pop(next(iter(tensors)))
    save_file(tensors, weight)
    with pytest.raises(ValueError, match="exactly match"):
        load_local_dino(tmp_path, sha256_file(weight), ch, family)
