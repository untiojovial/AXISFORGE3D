"""Explicit local-only frozen visual encoders; no implicit downloads."""
import hashlib
from pathlib import Path
import numpy as np
from research_grasping.phase3.vision_openclip import CHECKPOINT_SHA256
from research_grasping.representations.dino import load_local_dino, sha256_file
from .spatial import image_affine


def _images(images):
    if not isinstance(images,list) or not images: raise ValueError('nonempty image list required')
    for im in images:
        if not isinstance(im,np.ndarray) or im.dtype!=np.uint8 or im.ndim!=3 or im.shape[2]!=3: raise ValueError('expected RGB HWC uint8')
    if len({im.shape for im in images})!=1: raise ValueError('mixed image dimensions unsupported')


def _shortest_geometry(shape,shortest,crop):
    from transformers.image_transforms import get_resize_output_image_size
    h,w=shape[:2]; rh,rw=get_resize_output_image_size(np.zeros((h,w,3),dtype=np.uint8),shortest,default_to_square=False,input_data_format='channels_last')
    ch,cw=crop
    if rh<ch or rw<cw: raise ValueError('implicit padding not supported')
    top,left=(rh-ch)//2,(rw-cw)//2
    return {'origin_size':[h,w],'resized_size':[rh,rw], 'processed_size':[ch,cw], 'pixel_transform':image_affine((h,w),(rh,rw),(top,left)).tolist(), 'content_bounds':[-.5,-.5,ch-.5,cw-.5]}


class Frozen:
    def __init__(self, model, processor, kind, provenance, device):
        self.model=model.to(device).eval().requires_grad_(False); self.processor=processor; self.kind=kind; self.provenance=provenance; self.device=device
        if self.model.training or any(p.requires_grad for p in self.model.parameters()): raise ValueError('model not frozen')

    def encode(self, images):
        import torch
        from PIL import Image
        _images(images)
        if self.model.training or any(p.requires_grad for p in self.model.parameters()): raise ValueError('model no longer frozen')
        if self.kind=='openclip':
            steps=self.processor.transforms
            if ([type(v).__name__ for v in steps] != ['Resize','CenterCrop','MaybeConvertMode','MaybeToTensor','Normalize'] or steps[0].size != 224 or tuple(steps[1].size)!=(224,224) or steps[0].interpolation.value!='bicubic' or steps[0].antialias is not True):
                raise ValueError('unsupported OpenCLIP preprocessing geometry')
            x=torch.stack([self.processor(Image.fromarray(im)) for im in images]).to(self.device)
            h,w=images[0].shape[:2]
            if h!=w: raise ValueError('OpenCLIP geometry currently requires square input')
            geometry=_shortest_geometry(images[0].shape,224,(224,224))
            with torch.inference_mode():
                result=self.model.visual.forward_intermediates(x,indices=1,normalize_intermediates=True,intermediates_only=False)
                cls=torch.nn.functional.normalize(result['image_features'].float(),dim=-1)
                patches=result['image_intermediates'][0].permute(0,2,3,1)
        else:
            proc=self.processor
            if not (proc.do_resize and proc.do_center_crop and isinstance(proc.size,dict) and set(proc.size)=={'shortest_edge'} and isinstance(proc.crop_size,dict)):
                raise ValueError('unsupported processor geometry; no unverified mapping')
            crop=(proc.crop_size['height'],proc.crop_size['width'])
            geometry=_shortest_geometry(images[0].shape,proc.size['shortest_edge'],crop)
            batch=proc(images=[Image.fromarray(im) for im in images],return_tensors='pt')
            batch={k:v.to(self.device) for k,v in batch.items()}
            with torch.inference_mode(): output=self.model(**batch).last_hidden_state
            ph=self.model.config.patch_size
            if not isinstance(ph,int) or any(d%ph for d in crop): raise ValueError('invalid patch size')
            gh,gw=crop[0]//ph,crop[1]//ph
            registers=int(getattr(self.model.config,'num_register_tokens',0))
            if output.shape[1]!=1+registers+gh*gw: raise ValueError('unexpected token layout')
            cls=torch.nn.functional.normalize(output[:,0].float(),dim=-1)
            patches=output[:,1+registers:].reshape(len(images),gh,gw,-1)
        if tuple(x.shape[-2:]) != (224,224) if self.kind=='openclip' else tuple(batch['pixel_values'].shape[-2:]) != tuple(geometry['processed_size']): raise ValueError('processor size mismatch')
        if not torch.isfinite(cls).all() or not torch.isfinite(patches).all(): raise ValueError('nonfinite embeddings')
        gh,gw=patches.shape[1:3]; geometry['grid_shape']=[gh,gw]
        return {'global_embeddings':cls.cpu().numpy(),'patch_embeddings':patches.reshape(len(images),gh*gw,-1).cpu().numpy(),'token_geometry':geometry,'provenance':self.provenance}


def load_frozen(model_kind, checkpoint, config_sha256=None, weights_sha256=None, processor_sha256=None, device='cpu', model_revision=None):
    if model_kind=='openclip':
        import open_clip
        p=Path(checkpoint)
        if sha256_file(p)!=CHECKPOINT_SHA256 or (weights_sha256 and weights_sha256!=CHECKPOINT_SHA256): raise ValueError('checkpoint hash mismatch')
        model,_,processor=open_clip.create_model_and_transforms('ViT-B-32-quickgelu',pretrained=str(p),device=device,weights_only=True)
        provenance={'model_id':'open_clip/ViT-B-32-quickgelu/laion400m_e31','weights_sha256':CHECKPOINT_SHA256,'processor':str(processor),'open_clip_version':open_clip.__version__}
    elif model_kind in ('dinov2','dinov3'):
        import transformers
        from transformers import AutoImageProcessor
        p=Path(checkpoint)
        if not model_revision: raise ValueError('exact model revision required')
        if not all((config_sha256,weights_sha256,processor_sha256)): raise ValueError('all hashes are mandatory')
        if sha256_file(p/'preprocessor_config.json')!=processor_sha256: raise ValueError('processor hash mismatch')
        model=load_local_dino(p,weights_sha256,config_sha256,model_kind,device=device)
        processor=AutoImageProcessor.from_pretrained(str(p),local_files_only=True,trust_remote_code=False,use_fast=False)
        provenance={'model_id':model_kind,'config_sha256':config_sha256,'weights_sha256':weights_sha256,'processor_sha256':processor_sha256,'revision':model_revision,'transformers_version':transformers.__version__}
    else: raise ValueError('unsupported model')
    return Frozen(model,processor,model_kind,provenance,device)
