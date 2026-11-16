"""Final independent CPU64/MPS forward/backward check."""
import json
import torch
from .render import render_dense,Camera
from .tiled import render_tiled
from .scenes import make_scene
from .experiment import ROOT

def main():
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable')
    cpu=[x.requires_grad_() for x in make_scene(88,31,7,dtype=torch.float64)];gpu=[x.detach().float().to('mps').requires_grad_() for x in cpu];cam=Camera(37,29,35)
    ref=render_dense(*cpu,cam);got=render_tiled(*gpu,cam);a=torch.autograd.grad(ref.square().mean(),cpu);b=torch.autograd.grad(got.square().mean(),gpu)
    errors={name:float((x-y.detach().cpu().double()).abs().max()) for name,x,y in zip(['means','covariance','opacity','features'],a,b)}
    output=float((ref-got.detach().cpu().double()).abs().max())
    assert output<1e-5 and all(v<2e-4 for v in errors.values()),(output,errors)
    record={'device':'mps','forward_max_abs_error':output,'gradient_max_abs_error':errors,'torch':torch.__version__}
    (ROOT/'results/round05/mps_validation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
if __name__=='__main__':main()
