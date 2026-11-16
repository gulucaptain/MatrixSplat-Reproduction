import torch

def make_scene(seed=0,n=128,channels=3,device='cpu',dtype=torch.float32):
    generator=torch.Generator().manual_seed(seed)
    means=torch.rand(n,3,generator=generator,dtype=dtype);means[:,:2]=(means[:,:2]-.5)*2;means[:,2]=means[:,2]*2+2
    scales=torch.rand(n,3,generator=generator,dtype=dtype)*.09+.025
    covariance=torch.diag_embed(scales**2)
    opacity=torch.rand(n,generator=generator,dtype=dtype)*.6+.2
    features=torch.rand(n,channels,generator=generator,dtype=dtype)
    return tuple(x.to(device) for x in [means,covariance,opacity,features])
