"""Explicit local-only frozen visual encoders; no implicit downloads."""
import json
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
        elif self.kind=='dinov3-timm':
            # timm's DINOv3 copy: Resize(256, bicubic) + CenterCrop(256) + ImageNet normalisation.
            size=self.processor['size']
            if images[0].shape[0]!=images[0].shape[1]: raise ValueError('timm DINOv3 geometry currently requires square input')
            geometry=_shortest_geometry(images[0].shape,size,(size,size))
            x=torch.stack([self.processor['transform'](Image.fromarray(im)) for im in images]).to(self.device)
            with torch.inference_mode(): output=self.model.forward_features(x)
            ph=self.model.patch_embed.patch_size[0]; prefix=self.model.num_prefix_tokens
            if size%ph: raise ValueError('invalid patch size')
            gh=gw=size//ph
            if output.shape[1]!=prefix+gh*gw: raise ValueError('unexpected token layout')
            cls=torch.nn.functional.normalize(output[:,0].float(),dim=-1)
            patches=output[:,prefix:].reshape(len(images),gh,gw,-1)
        else:
            proc=self.processor
            size=proc.size
            original=images[0].shape[:2]
            if not proc.do_resize or not isinstance(size,dict):
                raise ValueError('unsupported processor geometry')
            if set(size)=={'height','width'} and not getattr(proc,'do_center_crop',False):
                crop=(size['height'],size['width'])
                geometry=dict(origin_size=list(original), resized_size=list(crop), processed_size=list(crop),
                              pixel_transform=image_affine(original,crop).tolist(),
                              content_bounds=[-.5,-.5,crop[0]-.5,crop[1]-.5])
            elif set(size)=={'shortest_edge'} and getattr(proc,'do_center_crop',False) and isinstance(proc.crop_size,dict):
                crop=(proc.crop_size['height'],proc.crop_size['width'])
                geometry=_shortest_geometry(images[0].shape,size['shortest_edge'],crop)
            else:
                raise ValueError('unsupported processor geometry; no unverified mapping')
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
        pixels=x if self.kind in ('openclip','dinov3-timm') else batch['pixel_values']
        if tuple(pixels.shape[-2:])!=tuple(geometry['processed_size']):
            raise ValueError('processor size mismatch')
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
        processor_type=json.loads((p/'preprocessor_config.json').read_text()).get('image_processor_type','')
        use_fast=processor_type.endswith('Fast') or processor_type=='DINOv3ViTImageProcessor'
        processor=AutoImageProcessor.from_pretrained(str(p),local_files_only=True,trust_remote_code=False,use_fast=use_fast)
        provenance={'model_id':model_kind,'config_sha256':config_sha256,'weights_sha256':weights_sha256,'processor_sha256':processor_sha256,'revision':model_revision,'transformers_version':transformers.__version__}
    elif model_kind=='dinov3-timm':
        # DINOv3 as redistributed by timm under the DINOv3 License (no access form).
        import timm
        import torch
        from safetensors.torch import load_file
        p=Path(checkpoint)
        if not model_revision: raise ValueError('exact model revision required')
        if not (config_sha256 and weights_sha256): raise ValueError('config and weight hashes are mandatory')
        if sha256_file(p/'config.json')!=config_sha256: raise ValueError('config hash mismatch')
        if sha256_file(p/'model.safetensors')!=weights_sha256: raise ValueError('checkpoint hash mismatch')
        config=json.loads((p/'config.json').read_text()); cfg=config['pretrained_cfg']
        if (config.get('architecture')!='vit_small_patch16_dinov3' or cfg.get('crop_pct')!=1.0
                or cfg.get('interpolation')!='bicubic' or cfg.get('input_size')!=[3,256,256]):
            raise ValueError('unsupported timm DINOv3 configuration')
        model=timm.create_model(config['architecture'],pretrained=False,num_classes=0)
        model.load_state_dict(load_file(str(p/'model.safetensors')),strict=True)
        # The original keeps RoPE periods in bfloat16; truncate to match its outputs (timm model card).
        model.rope.periods=model.rope.periods.to(torch.bfloat16).to(torch.float32)
        transform=timm.data.create_transform(input_size=tuple(cfg['input_size']),interpolation='bicubic',
                                             crop_pct=1.0,mean=tuple(cfg['mean']),std=tuple(cfg['std']))
        processor={'size':cfg['input_size'][1],'transform':transform}
        provenance={'model_id':'timm/vit_small_patch16_dinov3.lvd1689m','config_sha256':config_sha256,
                    'weights_sha256':weights_sha256,'revision':model_revision,'timm_version':timm.__version__,
                    'license':'DINOv3 License','rope_periods':'bfloat16-truncated','global':'CLS token after final norm'}
    else: raise ValueError('unsupported model')
    return Frozen(model,processor,model_kind,provenance,device)
