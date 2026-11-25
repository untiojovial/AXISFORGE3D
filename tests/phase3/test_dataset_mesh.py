import numpy as np
import pytest

trimesh = pytest.importorskip('trimesh')
from research_grasping.phase3.dataset_mesh import inspect_mesh


@pytest.mark.parametrize('extension', ['obj', 'stl', 'ply', 'glb'])
def test_mesh_formats_preserve_units_and_geometry(tmp_path, extension):
    original = trimesh.creation.box(extents=[20, 30, 40])
    path = tmp_path / ('box.' + extension)
    original.export(path)
    mesh, report = inspect_mesh(path, units='mm')
    np.testing.assert_allclose(mesh.extents, [0.02, 0.03, 0.04], rtol=1e-6)
    assert report['triangle_count'] == 12
    assert report['degenerate_faces'] == 0
    assert not report['scale_warning']


def test_explicit_units_and_scale_warning(tmp_path):
    path = tmp_path / 'box.stl'
    trimesh.creation.box().export(path)
    with pytest.raises(ValueError, match='Explicit units'):
        inspect_mesh(path, units='unknown')
    _, report = inspect_mesh(path, units='mm', min_extent_m=0.01)
    assert report['scale_warning']
