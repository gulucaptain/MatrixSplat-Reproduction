"""Import standard 3DGS PLY geometry/DC colors. Higher SH terms are explicitly unsupported."""
import numpy as np
import torch
from plyfile import PlyData

def load_ply(path,device='cpu'):
    p=PlyData.read(path)['vertex'];fields=p.data.dtype.names
    take=lambda names:np.stack([np.asarray(p[n]) for n in names],-1)
    means=take(['x','y','z']);scale=np.exp(take(['scale_0','scale_1','scale_2']));q=take(['rot_0','rot_1','rot_2','rot_3']);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=q.T;r=np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)
    cov=(r*scale[:,None,:]**2)@r.transpose(0,2,1)
    opacity=1/(1+np.exp(-np.asarray(p['opacity'])))
    rgb=np.clip(.5+.28209479177387814*take(['f_dc_0','f_dc_1','f_dc_2']),0,1)
    if any(k.startswith('f_rest_') for k in fields):
        import warnings;warnings.warn('Higher-order color SH terms are ignored by this DC-only importer',stacklevel=2)
    return tuple(torch.as_tensor(a,dtype=torch.float32,device=device) for a in [means,cov,opacity,rgb])
