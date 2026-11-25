import numpy as np
import pytest

from research_grasping.representations.cache import FeatureCache, cache_key
from research_grasping.representations.dino import load_local_dino, masked_pool
from research_grasping.representations.geometry import (
    backproject,
    pixel_to_patch,
    point_descriptors,
    project,
)


def test_depth_roundtrip_and_invalid():
    k = np.array([[2.0, 0, 1], [0, 2, 1], [0, 0, 1]])
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    depth = np.array([[1.0, 2.0], [np.nan, 0.0]])
    points, pixels = backproject(depth, k, t)
    uv, z = project(points, k, t)
    np.testing.assert_allclose(uv, pixels)
    np.testing.assert_allclose(z, [1, 2])
    np.testing.assert_allclose(points[0], [0.5, 1.5, 4])
    assert point_descriptors(points)["extent"].shape == (3,)
    with pytest.raises(ValueError):
        backproject(depth, k, np.zeros((4, 4)))


def test_patch_boundary_and_crop():
    p = np.array([[0, 0], [15, 0], [16, 0], [-1, 0], [31, 31], [32, 0]])
    np.testing.assert_array_equal(
        pixel_to_patch(p, np.eye(3), (32, 32), (32, 32), 16),
        [[0, 0], [0, 0], [1, 0], [-1, -1], [1, 1], [-1, -1]],
    )
    # Pixel centres after 2x resize transform as u' = 2(u+.5)-.5.
    a = np.array([[2, 0, 0.5], [0, 2, 0.5], [0, 0, 1.0]])
    np.testing.assert_array_equal(
        pixel_to_patch(np.array([[7.0, 0], [8.0, 0]]), a, (16, 16), (32, 32), 16),
        [[0, 0], [1, 0]],
    )


def test_cache_content_identity_and_safe_roundtrip(tmp_path):
    a = {"image": np.arange(5)}
    key = cache_key({"model": "one", "checkpoint": "abc"}, a)
    assert key != cache_key({"model": "two", "checkpoint": "abc"}, a)
    assert key != cache_key(
        {"model": "one", "checkpoint": "abc"}, {"image": np.arange(6)}
    )
    c = FeatureCache(tmp_path)
    assert c.get(key) is None
    c.put(key, {"features": np.arange(4)})
    np.testing.assert_array_equal(c.get(key)["features"], np.arange(4))
    with pytest.raises(ValueError):
        c.get("../escape")


def test_local_checkpoint_mandatory_and_empty_mask(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_local_dino(tmp_path, "0" * 64, "0" * 64, "dinov3")
    with pytest.raises(ValueError):
        masked_pool(np.ones((2, 2, 3)), np.zeros((2, 2)))
    np.testing.assert_array_equal(
        masked_pool(np.ones((2, 2, 3)), [[1, 0], [0, 0]]), np.ones(3)
    )


def test_crop_and_nondivisible_rejected():
    a = np.array([[1, 0, -16], [0, 1, 0], [0, 0, 1.0]])
    np.testing.assert_array_equal(
        pixel_to_patch(
            np.array([[0, 0], [16, 0], [47, 0], [48, 0]]), a, (32, 64), (32, 32), 16
        ),
        [[-1, -1], [0, 0], [1, 0], [-1, -1]],
    )
    with pytest.raises(ValueError):
        pixel_to_patch(np.zeros((1, 2)), np.eye(3), (32, 32), (31, 32), 16)


def test_token_layout_with_registers():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from research_grasping.representations.dino import extract_tokens

    class Model:
        def __init__(self, n):
            self.config = SimpleNamespace(patch_size=2, num_register_tokens=n)
            self.n = n

        def __call__(self, pixel_values):
            return SimpleNamespace(
                last_hidden_state=torch.arange((5 + self.n) * 3).reshape(
                    1, 5 + self.n, 3
                )
            )

    for n in (0, 2):
        cls, patches = extract_tokens(Model(n), torch.zeros((1, 3, 4, 4)))
        assert cls.shape == (1, 3) and patches.shape == (1, 2, 2, 3)
        assert patches[0, 0, 0, 0] == (1 + n) * 3
    with pytest.raises(ValueError):
        extract_tokens(Model(0), torch.zeros((1, 3, 3, 4)))
