"""Exploratory view and geometric similarities, development objects only."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True); a=p.parse_args()
    d=np.load(a.out/'summary.npz',allow_pickle=False)
    if not set(d['splits']).issubset({'train','validation'}): raise ValueError('sealed split found')
    manifest=json.loads((a.root/'manifest.json').read_text()); index={o['object_id']:o for o in manifest['objects'] if o['split'] in ('train','validation')}
    ids=d['ids']; dims=np.array([index[x]['dimensions_m'] for x in ids]); distances=np.linalg.norm(dims[:,None]-dims[None,:],axis=-1)
    ix=np.triu_indices(len(ids),1); rho=spearmanr(d['similarity_16'][ix],-distances[ix]).statistic
    np.savez_compressed(a.out/'geometry_comparison.npz',ids=ids,dimensions_m=dims,extent_distance_m=distances,spearman_visual_vs_negative_extent=rho)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,9)); image=ax.imshow(d['similarity_16'],vmin=-1,vmax=1,cmap='viridis'); ax.set_xticks(range(len(ids)),[str(x)[:25] for x in ids],rotation=90,fontsize=6); ax.set_yticks(range(len(ids)),[str(x)[:25] for x in ids],fontsize=6); ax.set_title('OpenCLIP cosine, 16 views (train/validation only)'); fig.colorbar(image,ax=ax); fig.tight_layout(); fig.savefig(a.out/'similarity_16.png',dpi=150); plt.close(fig)
    print('n',len(ids),'pairs',len(ix[0]),'spearman exploratory',rho)

if __name__=='__main__': main()
