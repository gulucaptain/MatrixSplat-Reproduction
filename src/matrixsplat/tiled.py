"""Apple GPU tile-batched formulation. No claim to reproduce NVIDIA MMA instructions."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .render import project

@dataclass
class TilePlan:
    indices:torch.Tensor
    valid:torch.Tensor
    points:torch.Tensor
    tile:int
    nx:int
    ny:int
    camera:object
    candidates:int

def make_plan(means,covariance,opacity,camera,tile=16,eps=1/255):
    with torch.no_grad():
        xy,_,cov,z=project(means,covariance,camera)
        xy=xy.detach().cpu().numpy();cov=cov.detach().cpu().numpy();op=opacity.detach().cpu().numpy();z=z.detach().cpu().numpy()
    radii=np.sqrt(2*np.maximum(0,np.log(np.maximum(op,eps)/eps))[:,None]*np.stack([cov[:,0,0],cov[:,1,1]],-1))
    lo=xy-radii;hi=xy+radii;nx=math.ceil(camera.width/tile);ny=math.ceil(camera.height/tile);lists=[];points=[]
    for iy in range(ny):
        for ix in range(nx):
            valid=(hi[:,0]>=ix*tile+.5)&(lo[:,0]<=(ix+1)*tile-.5)&(hi[:,1]>=iy*tile+.5)&(lo[:,1]<=(iy+1)*tile-.5)&(z>.01)&(op>=eps)
            lists.append(np.flatnonzero(valid))
            yy,xx=np.mgrid[iy*tile:(iy+1)*tile,ix*tile:(ix+1)*tile];points.append(np.stack([xx.ravel()+.5,yy.ravel()+.5],-1))
    k=max(1,max(map(len,lists)));idx=np.zeros((nx*ny,k),dtype=np.int64);mask=np.zeros_like(idx,dtype=bool)
    for i,lis in enumerate(lists):idx[i,:len(lis)]=lis;mask[i,:len(lis)]=True
    return TilePlan(torch.as_tensor(idx,device=means.device),torch.as_tensor(mask,device=means.device),torch.as_tensor(np.array(points),device=means.device,dtype=means.dtype),tile,nx,ny,camera,sum(map(len,lists)))

def render_tiled(means,covariance,opacity,features,camera,sigma=6.,visibility=None,bg_weight=.01,bg=None,plan=None,channel_batch=None,eps=1/255,half_accumulation=False):
    if plan is None:plan=make_plan(means,covariance,opacity,camera,eps=eps)
    if visibility is None:visibility=torch.ones_like(opacity)
    if bg is None:bg=torch.zeros(features.shape[1],device=features.device,dtype=features.dtype)
    xy,inv,cov,z=project(means,covariance,camera);idx=plan.indices
    d=plan.points[:,:,None,:]-xy[idx][:,None,:,:]
    conic=inv[idx];q=d[...,0]**2*conic[:,None,:,0,0]+2*d[...,0]*d[...,1]*conic[:,None,:,0,1]+d[...,1]**2*conic[:,None,:,1,1]
    alpha=(torch.exp(-.5*q)*opacity[idx][:,None,:]).clamp(max=.99)
    w=alpha*(1-z[idx][:,None,:]/sigma).clamp_min(0)*visibility[idx][:,None,:]
    w=torch.where(plan.valid[:,None,:]&(alpha>=eps)&(z[idx][:,None,:]>.01),w,0.)
    den=(w.sum(-1,keepdim=True)+bg_weight).clamp_min(1e-12)
    batch=channel_batch or features.shape[1];chunks=[]
    for c in range(0,features.shape[1],batch):
        chunk=features[:,c:c+batch][idx]
        product=torch.bmm(w.half(),chunk.half()).to(w.dtype) if half_accumulation else torch.bmm(w,chunk)
        chunks.append((product+bg_weight*bg[c:c+batch])/den)
    out=torch.cat(chunks,-1);out=out.reshape(plan.ny,plan.nx,plan.tile,plan.tile,-1).permute(0,2,1,3,4).reshape(plan.ny*plan.tile,plan.nx*plan.tile,-1)
    return out[:camera.height,:camera.width]
