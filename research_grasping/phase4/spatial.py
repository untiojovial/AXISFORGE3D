"""Map rectified image pixels to token grids after explicit image transforms."""
import numpy as np
from research_grasping.representations.geometry import project


def image_affine(origin_size, resized_size, crop_origin=(0,0), pad_before=(0,0)):
    """Return affine x,y with half-pixel resize; crop/pad tuples are (top,left)."""
    h,w=origin_size; rh,rw=resized_size; top,left=crop_origin; pt,pl=pad_before
    if min(h,w,rh,rw)<=0 or min(top,left,pt,pl)<0: raise ValueError('invalid image transform')
    sx,sy=rw/w,rh/h
    return np.array([[sx,0,(sx-1)/2-left+pl],[0,sy,(sy-1)/2-top+pt],[0,0,1.]])


def visible_patch(point_world, depth, k, world_from_camera, patches, *, pixel_transform=None, processed_size=None, content_bounds=None, tolerance=0.003):
    """Visibility in original depth followed by explicit processed-image mapping.

    content_bounds=(top,left,bottom,right) identifies real image content in output;
    boundaries are pixel-edge coordinates, e.g. (-.5,-.5,H-.5,W-.5).
    No support for distortion or non-affine preprocessing.
    """
    depth=np.asarray(depth); patches=np.asarray(patches)
    if pixel_transform is None and processed_size is None and content_bounds is None:
        if depth.ndim!=2 or depth.shape[0]!=depth.shape[1] or patches.ndim!=3 or patches.shape[:2]!=(7,7): raise ValueError('explicit token geometry required')
        pixel_transform=image_affine(depth.shape,(224,224)); processed_size=(224,224); content_bounds=(-.5,-.5,223.5,223.5)
    transform=np.asarray(pixel_transform,float)
    if depth.ndim!=2 or patches.ndim!=3 or min(patches.shape[:2])<=0 or transform.shape!=(3,3) or not np.isfinite(transform).all() or not np.allclose(transform[2],[0,0,1]) or not np.isfinite(tolerance) or tolerance<0:
        raise ValueError('invalid inputs')
    h,w=processed_size; gh,gw=patches.shape[:2]
    if min(h,w)<=0 or h%gh or w%gw: raise ValueError('incompatible token grid')
    bounds=np.asarray(content_bounds,float)
    if bounds.shape!=(4,) or not np.isfinite(bounds).all() or bounds[0]>=bounds[2] or bounds[1]>=bounds[3]: raise ValueError('invalid content bounds')
    point=np.asarray(point_world,float)
    if point.shape!=(3,): raise ValueError('invalid 3D point')
    uv,z=project(point[None],k,world_from_camera)
    if z[0]<=0 or not np.isfinite(uv).all(): return None
    u,v=uv[0]; oh,ow=depth.shape
    if u<-.5 or v<-.5 or u>=ow-.5 or v>=oh-.5: return None
    x,y=int(np.floor(u+.5)),int(np.floor(v+.5))
    if not np.isfinite(depth[y,x]) or depth[y,x]<=0 or abs(depth[y,x]-z[0])>tolerance: return None
    q=transform@np.array([u,v,1.]); x,y=q[:2]
    if x<-.5 or x>=w-.5 or y<-.5 or y>=h-.5 or y<bounds[0] or y>=bounds[2] or x<bounds[1] or x>=bounds[3]: return None
    px=int(np.floor((x+.5)*gw/w)); py=int(np.floor((y+.5)*gh/h))
    return patches[py,px]
