"""Adversarial archive and tensor inputs; all values are simulated test data."""
import io
import stat
import zipfile
import numpy as np
import pytest
from research_grasping.phase4.bundle_io import safe_members, validate_npz


def npz(**arrays):
    buf=io.BytesIO(); np.savez(buf,**arrays); return buf.getvalue()


@pytest.mark.parametrize('global_vectors,ids,patches,grid,error',[
    (np.ones((15,4)),np.arange(16),None,None,'global'),
    (np.full((16,4),np.nan),np.arange(16),None,None,'global'),
    (np.ones((16,4),dtype=np.int32),np.arange(16),None,None,'global'),
    (np.ones((16,4)),np.arange(16)[::-1],None,None,'view IDs'),
    (np.ones((16,4)),np.arange(16,dtype=float),None,None,'view IDs'),
    (np.ones((16,4)),np.arange(16),np.full((16,2,3),np.inf),[1,2],'patch embeddings'),
    (np.ones((16,4)),np.arange(16),np.ones((16,2,3)),[2,2],'patch grid'),
])
def test_reject_invalid_npz(global_vectors,ids,patches,grid,error):
    arrays={'global_embeddings':global_vectors,'view_ids':ids}
    if patches is not None: arrays['patch_embeddings']=patches
    with pytest.raises(ValueError,match=error):
        validate_npz(npz(**arrays),token_geometry=None if grid is None else {'grid_shape':grid})


def test_missing_extra_and_unsafe_numpy_members():
    with pytest.raises(ValueError,match='invalid NPZ members'): validate_npz(npz(view_ids=np.arange(16)))
    with pytest.raises(ValueError,match='invalid NPZ members'):
        validate_npz(npz(view_ids=np.arange(16),global_embeddings=np.ones((16,2)),extra=np.ones(1)))
    with pytest.raises(ValueError,match='unsafe array header'):
        validate_npz(npz(view_ids=np.arange(16),global_embeddings=np.array([object()],dtype=object)))


@pytest.mark.parametrize('names',['../x','/root/x','a/./b','a//b','a\\b','folder/'])
def test_reject_paths_aliases_and_directories(names):
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w') as z: z.writestr(names,b'x')
    with zipfile.ZipFile(io.BytesIO(mem.getvalue())) as z:
        with pytest.raises(ValueError,match='unsafe'): safe_members(z)


def test_reject_duplicate_names_and_symlink():
    mem=io.BytesIO()
    with pytest.warns(UserWarning,match='Duplicate'):
        with zipfile.ZipFile(mem,'w') as z: z.writestr('same',b'a');z.writestr('same',b'b')
    with zipfile.ZipFile(io.BytesIO(mem.getvalue())) as z:
        with pytest.raises(ValueError,match='unsafe'): safe_members(z)
    mem=io.BytesIO(); link=zipfile.ZipInfo('link');link.create_system=3;link.external_attr=(stat.S_IFLNK|0o777)<<16
    with zipfile.ZipFile(mem,'w') as z:z.writestr(link,'target')
    with zipfile.ZipFile(io.BytesIO(mem.getvalue())) as z:
        with pytest.raises(ValueError,match='unsafe'):safe_members(z)


def test_archive_size_limit(monkeypatch):
    import research_grasping.phase4.bundle_io as module
    monkeypatch.setattr(module,'MAX_FILES',0)
    mem=io.BytesIO()
    with zipfile.ZipFile(mem,'w') as z:z.writestr('x','a')
    with zipfile.ZipFile(io.BytesIO(mem.getvalue())) as z:
        with pytest.raises(ValueError,match='resource limits'):safe_members(z)
