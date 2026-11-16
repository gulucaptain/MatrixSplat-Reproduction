from dataclasses import dataclass
import torch

@dataclass
class Camera:
    width:int=64
    height:int=64
    focal:float=60.
    cx:float|None=None
    cy:float|None=None


def project(means,covariances,camera):
    """Pinhole Jacobian projection. Inputs are in camera coordinates (+Z forward)."""
    x,y,z=means.unbind(-1); safe_z=z.clamp_min(1e-5);f=camera.focal
    u=f*x/safe_z+(camera.cx if camera.cx is not None else camera.width/2)
    v=f*y/safe_z+(camera.cy if camera.cy is not None else camera.height/2)
    zero=torch.zeros_like(z)
    j=torch.stack([f/safe_z,zero,-f*x/safe_z**2,zero,f/safe_z,-f*y/safe_z**2],-1).reshape(-1,2,3)
    cov=j@covariances@j.transpose(-1,-2)+torch.eye(2,device=z.device,dtype=z.dtype)*.3
    # explicit 2x2 inverse supported on CPU and MPS
    a,b,c=cov[:,0,0],cov[:,0,1],cov[:,1,1];det=(a*c-b*b).clamp_min(1e-12)
    inv=torch.stack([c,-b,-b,a],-1).reshape(-1,2,2)/det[:,None,None]
    return torch.stack([u,v],-1),inv,cov,z


def pixels(camera,device,dtype):
    y,x=torch.meshgrid(torch.arange(camera.height,device=device,dtype=dtype)+.5,torch.arange(camera.width,device=device,dtype=dtype)+.5,indexing='ij')
    return torch.stack([x.flatten(),y.flatten()],-1)


def alpha_matrix(xy,inv,depth,opacity,points,eps=1/255):
    d=points[:,None,:]-xy[None,:,:]
    q=(d*(d@torch.eye(2,device=d.device,dtype=d.dtype))).sum(-1) if False else torch.einsum('pni,nij,pnj->pn',d,inv,d)
    alpha=(torch.exp(-.5*q)*opacity[None,:]).clamp(max=.99)
    return torch.where((alpha>=eps)&(depth[None,:]>.01),alpha,0.)


def weights(alpha,depth,sigma,visibility):
    return alpha*(1-depth/sigma).clamp_min(0)[None,:]*visibility[None,:]


def blend_reference(w,features,bg_weight,bg):
    """Independent scalar Gaussian accumulation for validation, not a speed baseline."""
    numerator=torch.zeros((w.shape[0],features.shape[1]),dtype=w.dtype,device=w.device)+bg_weight*bg
    denominator=torch.zeros((w.shape[0],1),dtype=w.dtype,device=w.device)+bg_weight
    for i in range(w.shape[1]):
        numerator=numerator+w[:,i:i+1]*features[i]
        denominator=denominator+w[:,i:i+1]
    return numerator/denominator.clamp_min(1e-12)


def blend_matrix(w,features,bg_weight,bg,channel_batch=None):
    den=(w.sum(-1,keepdim=True)+bg_weight).clamp_min(1e-12)
    if channel_batch:
        return torch.cat([(w@features[:,i:i+channel_batch]+bg_weight*bg[i:i+channel_batch])/den for i in range(0,features.shape[1],channel_batch)],-1)
    return (w@features+bg_weight*bg)/den


def blend_alpha(alpha,depth,features,bg):
    """Vectorized exact depth-ordered compositing with background; fair tensor baseline."""
    order=torch.argsort(depth);a=alpha[:,order]
    trans=torch.cumprod(torch.cat([torch.ones_like(a[:,:1]),1-a],-1),-1)
    return (a*trans[:,:-1])@features[order]+trans[:,-1:]*bg


def render_dense(means,covariances,opacity,features,camera,sigma=6.,visibility=None,bg_weight=.01,bg=None,mode='matrix',channel_batch=None):
    if visibility is None:visibility=torch.ones_like(opacity)
    if bg is None:bg=torch.zeros(features.shape[1],device=features.device,dtype=features.dtype)
    xy,inv,cov,z=project(means,covariances,camera)
    alpha=alpha_matrix(xy,inv,z,opacity,pixels(camera,means.device,means.dtype))
    if mode=='alpha':out=blend_alpha(alpha,z,features,bg)
    else:
        w=weights(alpha,z,sigma,visibility)
        out=blend_reference(w,features,bg_weight,bg) if mode=='reference' else blend_matrix(w,features,bg_weight,bg,channel_batch)
    return out.reshape(camera.height,camera.width,-1)
