"""Multiview synthetic regression: fixed geometry, learned blending parameters.
Not the paper's dataset/training protocol. Target renderer is depth-ordered alpha.
"""
import argparse,json,time,math,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image
from .scenes import make_scene
from .render import Camera,render_dense
from .tiled import make_plan,render_tiled
from .experiment import ROOT,synchronize

class Appearance(torch.nn.Module):
    def __init__(self,opacity,features):
        super().__init__();self.logits=torch.nn.Parameter(torch.logit(features.clamp(.001,.999)));self.opacity=torch.nn.Parameter(torch.logit(opacity))
        self.visibility=torch.nn.Parameter(torch.zeros(len(opacity),4,device=opacity.device));self.visibility.data[:,0]=math.log(math.expm1(1.))
        self.scale=torch.nn.Parameter(torch.tensor(math.log(math.expm1(6.)),device=opacity.device));self.background=torch.nn.Parameter(torch.tensor(math.log(math.expm1(.01)),device=opacity.device))
    def values(self,means):
        # Positive degree-one directional basis; documented adaptation, not claimed exact author parameterization.
        directions=F.normalize(means,dim=-1);basis=torch.cat([torch.ones_like(means[:,:1]),directions],-1)
        return self.opacity.sigmoid(),self.logits.sigmoid(),F.softplus((self.visibility*basis).sum(-1)),F.softplus(self.scale)+.01,F.softplus(self.background)+1e-6


def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=5);p.add_argument('--steps',type=int,default=240);p.add_argument('--device',default='mps');a=p.parse_args()
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.set_num_threads(8);out=ROOT/'results/round04';out.mkdir(parents=True,exist_ok=True);records=[];start=time.perf_counter()
    for seed in range(a.seeds):
        means,cov,op,rgb=make_scene(seed,72,3,a.device)
        labels=(means[:,0]>.3).long()+(means[:,0]>-.3).long();features=torch.cat([rgb,F.one_hot(labels,3).float()],-1)
        cam=Camera(48,48,50);views=[];targets=[];plans=[]
        for shift in [-.3,0,.3,.15]:
            m=means.clone();m[:,0]+=shift;views.append(m)
            with torch.no_grad():targets.append(render_dense(m,cov,op,features,cam,mode='alpha'))
            # Conservative plan remains valid as opacity learns up to 1; geometry is fixed.
            plans.append(make_plan(m,cov,torch.ones_like(op),cam))
        model=Appearance(op,features);opt=torch.optim.Adam(model.parameters(),lr=.025);history=[]
        def predict(view):
            opacity,feat,vis,sigma,bg=model.values(views[view]);return render_tiled(views[view],cov,opacity,feat,cam,visibility=vis,sigma=sigma,bg_weight=bg,plan=plans[view])
        with torch.no_grad():before=[predict(v).detach() for v in range(4)]
        for step in range(a.steps):
            v=step%3;opt.zero_grad(set_to_none=True);pred=predict(v);loss=(pred-targets[v]).square().mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            if step%20==0 or step==a.steps-1:
                history.append({'step':step,'view':v,'loss':float(loss.detach())});print(f'seed={seed} step={step} loss={float(loss.detach()):.7f}',flush=True)
        with torch.no_grad():after=[predict(v).detach() for v in range(4)]
        psnr=lambda x,y:float(-10*torch.log10((x[...,:3]-y[...,:3]).square().mean().clamp_min(1e-12)))
        row={'seed':seed,'train_psnr_before':np.mean([psnr(before[v],targets[v]) for v in range(3)]),'train_psnr_after':np.mean([psnr(after[v],targets[v]) for v in range(3)]),'heldout_psnr_before':psnr(before[3],targets[3]),'heldout_psnr_after':psnr(after[3],targets[3]),'heldout_feature_mse':float((after[3][...,3:]-targets[3][...,3:]).square().mean()),'history':history};records.append(row)
        torch.save({'appearance':{k:v.detach().cpu() for k,v in model.state_dict().items()},'means':means.cpu(),'covariance':cov.cpu(),'opacity':op.cpu(),'features':features.cpu(),'seed':seed,'camera':vars(cam)},out/f'seed{seed:02d}.pt')
        panels=[targets[3][...,:3],before[3][...,:3],after[3][...,:3]];arr=torch.cat(panels,1).clamp(0,1).cpu().numpy();Image.fromarray((arr*255).astype('uint8')).resize((864,288)).save(out/f'seed{seed:02d}_heldout.png')
        (out/'raw.json').write_text(json.dumps(records,indent=2))
    synchronize(a.device);(out/'manifest.json').write_text(json.dumps({'args':vars(a),'elapsed_seconds':time.perf_counter()-start,'torch':torch.__version__,'scope':'synthetic fixed geometry; appearance/opacity/visibility/scale/background training only','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
if __name__=='__main__':main()
