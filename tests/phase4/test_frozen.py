"""CPU architecture smoke tests: random weights, no pretrained claims."""
import numpy as np
import pytest
import hashlib
from pathlib import Path

torch=pytest.importorskip('torch'); tr=pytest.importorskip('transformers')
from research_grasping.phase4.frozen_models import load_frozen
from research_grasping.representations.dino import sha256_file


def state(model):
    h=hashlib.sha256()
    for name,p in model.state_dict().items(): h.update(name.encode());h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


@pytest.mark.parametrize('kind',['dinov2','dinov3'])
def test_frozen_dino_offline_random_architecture(tmp_path,kind,monkeypatch):
    args=dict(image_size=28,patch_size=7,hidden_size=32,num_hidden_layers=1,num_attention_heads=4,intermediate_size=64)
    model=tr.Dinov2Model(tr.Dinov2Config(**args)) if kind=='dinov2' else tr.DINOv3ViTModel(tr.DINOv3ViTConfig(**args,num_register_tokens=2))
    model.save_pretrained(tmp_path,safe_serialization=True)
    tr.BitImageProcessor(size={'shortest_edge':32},crop_size={'height':28,'width':28}).save_pretrained(tmp_path)
    weights=sha256_file(tmp_path/'model.safetensors'); config=sha256_file(tmp_path/'config.json'); processor=sha256_file(tmp_path/'preprocessor_config.json')
    def no_network(*args,**kwargs): raise AssertionError('network access attempted')
    import socket
    monkeypatch.setattr(socket.socket,'connect',no_network)
    f=load_frozen(kind,tmp_path,config,weights,processor,model_revision='test-random')
    before=state(f.model); processor_before=f.processor.to_json_string(); out=f.encode([np.zeros((32,40,3),np.uint8)])
    assert out['global_embeddings'].shape==(1,32) and out['patch_embeddings'].shape==(1,16,32)
    geom=out['token_geometry']; assert geom['resized_size']==[32,40] and geom['processed_size']==[28,28]
    np.testing.assert_allclose(geom['pixel_transform'],[[1,0,-6],[0,1,-2],[0,0,1]])
    assert f.processor.to_json_string()==processor_before
    assert state(f.model)==before and not f.model.training and all(not p.requires_grad for p in f.model.parameters())
    with pytest.raises(ValueError): load_frozen(kind,tmp_path,config,weights,'0'*64,model_revision='test-random')


def test_pretrained_openclip_frozen_repeat_if_available(monkeypatch):
    import os, socket
    path=os.getenv('AXISFORGE_OPENCLIP_CHECKPOINT')
    if not path: pytest.skip('set AXISFORGE_OPENCLIP_CHECKPOINT to authorized local file')
    monkeypatch.setattr(socket.socket,'connect',lambda *a,**kw: (_ for _ in ()).throw(AssertionError('network attempted')))
    torch.set_num_threads(4)
    f=load_frozen('openclip',path); before=state(f.model)
    rgb=np.arange(96*96*3,dtype=np.uint8).reshape(96,96,3)
    a=f.encode([rgb]); b=f.encode([rgb])
    np.testing.assert_allclose(a['global_embeddings'],b['global_embeddings'],rtol=0,atol=1e-7)
    np.testing.assert_allclose(a['patch_embeddings'],b['patch_embeddings'],rtol=0,atol=1e-7)
    assert state(f.model)==before and not f.model.training and not any(p.requires_grad for p in f.model.parameters())


def test_input_contract_and_checkpoint_rejection(tmp_path):
    from research_grasping.phase4.frozen_models import _images, _shortest_geometry
    with pytest.raises(ValueError): _images([])
    with pytest.raises(ValueError): _images([np.zeros((2,2),np.uint8)])
    with pytest.raises(ValueError): _images([np.zeros((2,2,3),np.float32)])
    with pytest.raises(ValueError): _images([np.zeros((2,2,3),np.uint8),np.zeros((3,2,3),np.uint8)])
    with pytest.raises(ValueError): _shortest_geometry((4,4,3),4,(8,8))
    with pytest.raises(ValueError): load_frozen('unrecognized',tmp_path)
    with pytest.raises(ValueError): load_frozen('dinov2',tmp_path)
    with pytest.raises(ValueError): load_frozen('dinov2',tmp_path,model_revision='revision')
    invalid=tmp_path/'bad.pt';invalid.write_bytes(b'invalid')
    with pytest.raises(ValueError): load_frozen('openclip',invalid)


def test_refuses_unfrozen_model_and_unsupported_geometry():
    from research_grasping.phase4.frozen_models import Frozen
    class Dummy(torch.nn.Module):
        def __init__(self): super().__init__(); self.w=torch.nn.Parameter(torch.ones(1))
    processor=tr.BitImageProcessor(size={'shortest_edge':32},crop_size={'height':28,'width':28})
    f=Frozen(Dummy(),processor,'dinov2',{},'cpu'); image=np.zeros((32,32,3),np.uint8)
    f.model.train()
    with pytest.raises(ValueError,match='no longer frozen'): f.encode([image])
    f.model.eval(); f.model.w.requires_grad_(True)
    with pytest.raises(ValueError,match='no longer frozen'): f.encode([image])
    f.model.requires_grad_(False); processor.do_resize=False
    with pytest.raises(ValueError,match='unsupported processor'): f.encode([image])
