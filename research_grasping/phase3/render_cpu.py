"""Deterministic CPU triangle z-buffer; metric rectified camera, no culling."""
import numpy as np
from research_grasping.representations.geometry import backproject, _calibration


def look_at(position, target):
    """Camera x right, y down, z forward; world +z is nominal up."""
    pos = np.asarray(position, float); target = np.asarray(target, float)
    forward = target-pos
    if np.linalg.norm(forward) < 1e-12: raise ValueError('invalid camera')
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0,0,1.])
    if np.linalg.norm(right) < 1e-12: raise ValueError('camera parallel to up')
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    t = np.eye(4); t[:3,:3] = np.column_stack((right, down, forward)); t[:3,3] = pos
    return t


def cameras(bounds, count, width=128, height=128):
    """Fixed azimuth orbit, deterministic constant 25° elevation, 1/4/8/16 views."""
    if count not in (1,4,8,16): raise ValueError('unsupported view count')
    b=np.asarray(bounds,float)
    if b.shape != (2,3) or not np.isfinite(b).all() or np.any(b[1] <= b[0]): raise ValueError('invalid bounds')
    centre=b.mean(0); radius=np.linalg.norm(b[1]-b[0])*1.5
    k=np.array([[width,0,(width-1)/2],[0,width,(height-1)/2],[0,0,1.]])
    transforms=[]
    for a in 2*np.pi*np.arange(count)/count:
        direction=np.array([np.cos(a)*np.cos(np.deg2rad(25)),np.sin(a)*np.cos(np.deg2rad(25)),np.sin(np.deg2rad(25))])
        transforms.append(look_at(centre+radius*direction,centre))
    return k, transforms


def render(vertices, faces, k, world_from_camera, height, width):
    """Flat face normals, gray Lambert-like shading, axial z; NaN background.

    No shadows, texture, backface culling, near-plane clipping or transparency.
    Triangles crossing the near plane are rejected, not clipped.
    """
    k,t=_calibration(k,world_from_camera)
    vertices=np.asarray(vertices,float); faces=np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3 or not np.issubdtype(faces.dtype,np.integer) or len(faces)==0 or np.any(faces<0) or np.any(faces>=len(vertices)) or height<=0 or width<=0:
        raise ValueError('invalid mesh or image')
    camera=(vertices-t[:3,3]) @ t[:3,:3]
    if np.any(camera[:,2] <= 0): raise ValueError('near-plane clipping unsupported')
    p=camera @ k.T; uv=p[:,:2]/p[:,2,None]
    depth=np.full((height,width),np.inf); normals=np.full((height,width,3),np.nan)
    for face in faces:
        v=uv[face]; z=camera[face,2]
        lo=np.maximum(np.ceil(v.min(axis=0)).astype(int),0); hi=np.minimum(np.floor(v.max(axis=0)).astype(int),[width-1,height-1])
        if np.any(lo>hi): continue
        a,b,c=v; den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(den)<1e-12: continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        wa=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/den
        wb=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/den
        wc=1-wa-wb; inside=(wa>=-1e-10)&(wb>=-1e-10)&(wc>=-1e-10)
        inv=wa/z[0]+wb/z[1]+wc/z[2]; candidate=np.divide(1,inv,out=np.full_like(inv,np.inf),where=inv>0)
        sub=depth[lo[1]:hi[1]+1,lo[0]:hi[0]+1]; update=inside & (candidate<sub)
        n=np.cross(vertices[face[1]]-vertices[face[0]],vertices[face[2]]-vertices[face[0]])
        norm=np.linalg.norm(n)
        if norm<1e-15: continue
        sub[update]=candidate[update]
        normals[lo[1]:hi[1]+1,lo[0]:hi[0]+1][update]=n/norm
    mask=np.isfinite(depth); depth[~mask]=np.nan
    rgb=np.zeros((height,width,3),np.uint8)
    # Two-sided shading, not an estimate of true illumination.
    intensity=np.maximum(.25,np.abs(np.nansum(normals*t[:3,2],axis=-1)))*190
    rgb[mask]=intensity[mask,None].astype(np.uint8)
    points,pixels=backproject(depth,k,t)
    return {'rgb':rgb,'depth':depth,'mask':mask,'normals_world':normals,'points_world':points,'pixels':pixels,'K':k,'world_from_camera':t}
