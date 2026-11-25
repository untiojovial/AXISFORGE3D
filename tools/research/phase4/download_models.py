"""Explicit download from official HF model IDs; no license acceptance or token output."""
import argparse, hashlib, json, os, re
from pathlib import Path

IDS={'dinov2':'facebook/dinov2-small','dinov3':'facebook/dinov3-vits16-pretrain-lvd1689m'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=IDS,required=True);p.add_argument('--revision',required=True,help='full 40-character official commit SHA')
    p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    if not re.fullmatch(r'[0-9a-f]{40}',a.revision): p.error('revision must be an exact 40-character hexadecimal commit')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=IDS[a.model],revision=a.revision,local_dir=a.out,token=os.environ.get('HF_TOKEN'),allow_patterns=['config.json','preprocessor_config.json','model.safetensors','README.md','LICENSE*'])
    hashes={}
    for name in ('config.json','preprocessor_config.json','model.safetensors'):
        f=a.out/name
        if not f.is_file(): raise FileNotFoundError(f'missing {name}')
        h=hashlib.sha256()
        with f.open('rb') as source:
            for block in iter(lambda:source.read(1024*1024),b''): h.update(block)
        hashes[name]=h.hexdigest()
    print(json.dumps({'model_id':IDS[a.model],'revision':a.revision,'sha256':hashes},sort_keys=True))

if __name__=='__main__': main()
