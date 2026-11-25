"""Inspect the versioned UR5 base collision mesh as STL triangle vertices."""

import hashlib
import struct
from pathlib import Path

import numpy as np

from research_grasping.representations.geometry import point_descriptors

path = Path("apps/viewer/samples/ur5/meshes/base.stl")
data = path.read_bytes()
triangles = struct.unpack_from("<I", data, 80)[0]
if len(data) != 84 + 50 * triangles:
    raise ValueError("not the expected binary STL format")
vertices = np.array(
    [struct.unpack_from("<9f", data, 84 + 50 * i + 12) for i in range(triangles)]
).reshape(-1, 3)
features = point_descriptors(vertices)
print("sha256", hashlib.sha256(data).hexdigest())
print("triangles", triangles, "vertex occurrences", len(vertices))
print("extent", features["extent"])
print("centroid", features["centroid"])
