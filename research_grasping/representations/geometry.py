"""Rectified, registered RGB-D geometry. Pixel centres have integer coordinates."""

import numpy as np


def _calibration(k, transform):
    k = np.asarray(k, dtype=float)
    transform = np.asarray(transform, dtype=float)
    if (
        k.shape != (3, 3)
        or not np.isfinite(k).all()
        or not np.allclose(k[2], [0, 0, 1])
        or np.linalg.det(k) <= 0
    ):
        raise ValueError("invalid intrinsic matrix")
    if (
        transform.shape != (4, 4)
        or not np.isfinite(transform).all()
        or not np.allclose(transform[3], [0, 0, 0, 1])
    ):
        raise ValueError("invalid transform")
    r = transform[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(r), 1, atol=1e-6
    ):
        raise ValueError("transform must be rigid")
    return k, transform


def backproject(depth, k, world_from_camera, max_depth=np.inf):
    """Return valid (world points, pixel coordinates); depth is axial metres."""
    k, t = _calibration(k, world_from_camera)
    d = np.asarray(depth, dtype=float)
    if d.ndim != 2 or not (max_depth > 0):
        raise ValueError("invalid depth or range")
    v, u = np.nonzero(np.isfinite(d) & (d > 0) & (d <= max_depth))
    pixels = np.column_stack((u, v))
    rays = np.linalg.solve(k, np.column_stack((pixels, np.ones(len(u)))).T).T
    camera = rays * d[v, u, None]
    return camera @ t[:3, :3].T + t[:3, 3], pixels


def project(points, k, world_from_camera):
    """Project world points, returning pixel coordinates and axial depth; behind-camera points are NaN."""
    k, t = _calibration(k, world_from_camera)
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all():
        raise ValueError("expected finite Nx3 points")
    cam = (p - t[:3, 3]) @ t[:3, :3]
    homogeneous = cam @ k.T
    uv = np.full((len(p), 2), np.nan)
    valid = cam[:, 2] > 0
    uv[valid] = homogeneous[valid, :2] / homogeneous[valid, 2, None]
    return uv, cam[:, 2]


def pixel_to_patch(pixels, pixel_transform, origin_size, image_size, patch_size):
    """Map pixel centres through affine preprocessing; cropped/padded pixels return (-1,-1).

    image_size=(height,width) is the unpadded transformed content; grid origin
    is top-left. Pixel edges are half a pixel from their integer centres.
    """
    p = np.asarray(pixels, dtype=float)
    a = np.asarray(pixel_transform, dtype=float)
    h, w = image_size
    oh, ow = origin_size
    if (
        p.ndim != 2
        or p.shape[1] != 2
        or not np.isfinite(p).all()
        or a.shape != (3, 3)
        or not np.isfinite(a).all()
        or not np.allclose(a[2], [0, 0, 1])
        or h <= 0
        or w <= 0
        or oh <= 0
        or ow <= 0
        or patch_size <= 0
        or h % patch_size
        or w % patch_size
    ):
        raise ValueError("invalid pixel mapping")
    q = np.column_stack((p, np.ones(len(p)))) @ a.T
    ok = (
        (p[:, 0] >= -0.5)
        & (p[:, 0] < ow - 0.5)
        & (p[:, 1] >= -0.5)
        & (p[:, 1] < oh - 0.5)
        & (q[:, 0] >= -0.5)
        & (q[:, 0] < w - 0.5)
        & (q[:, 1] >= -0.5)
        & (q[:, 1] < h - 0.5)
    )
    result = np.full((len(p), 2), -1, dtype=int)
    result[ok] = np.floor((q[ok, :2] + 0.5) / patch_size).astype(int)
    return result


def point_descriptors(points):
    """View-dependent box extents, centroid and covariance in input units."""
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("expected at least two finite 3D points")
    return {
        "extent": np.ptp(p, axis=0),
        "centroid": p.mean(axis=0),
        "covariance": np.cov(p, rowvar=False),
    }
