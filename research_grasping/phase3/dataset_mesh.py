"""Inspect externally supplied meshes without inferring units or repairing them."""
from pathlib import Path
import numpy as np

_UNIT_TO_M = {'m': 1.0, 'cm': 0.01, 'mm': 0.001}
_FORMATS = {'.obj', '.stl', '.ply', '.glb'}


def inspect_mesh(path, *, units, min_extent_m=1e-4, max_extent_m=2.0):
    """Load geometry in SI; bounds are warnings, not proof of correct scale.

    Scene-node transforms are applied. The returned mesh is a separate object,
    has no automatic repair and preserves the source axes and origin.
    """
    import trimesh

    path = Path(path)
    if path.suffix.lower() not in _FORMATS:
        raise ValueError('Supported formats: OBJ, STL, PLY, GLB')
    if units not in _UNIT_TO_M:
        raise ValueError('Explicit units required: m, cm or mm')
    if not 0 < min_extent_m < max_extent_m:
        raise ValueError('Invalid scale limits')
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        # Dump applies each node transform. No per-component repair occurs.
        parts = loaded.dump()
        if len(parts) == 0:
            raise ValueError('Empty scene')
        mesh = trimesh.util.concatenate(parts)
    else:
        mesh = loaded.copy()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError('Mesh must contain vertices and triangles')
    mesh.apply_scale(_UNIT_TO_M[units])
    vertices = np.asarray(mesh.vertices)
    if not np.isfinite(vertices).all():
        raise ValueError('Non-finite vertex coordinates')
    areas = np.asarray(mesh.area_faces)
    if not np.isfinite(areas).all():
        raise ValueError('Non-finite face area')
    extents = np.asarray(mesh.extents)
    largest = float(extents.max())
    return mesh, {
        'source_units': units,
        'scale_to_m': _UNIT_TO_M[units],
        'dimensions_m': extents.tolist(),
        'bounds_m': np.asarray(mesh.bounds).tolist(),
        'triangle_count': len(mesh.faces),
        'degenerate_faces': int(np.count_nonzero(areas <= 1e-18)),
        'watertight': bool(mesh.is_watertight),
        'scale_warning': bool(largest < min_extent_m or largest > max_extent_m),
        'volume_m3': float(mesh.volume) if mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0 else None,
    }
