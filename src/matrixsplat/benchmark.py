import argparse,json,random,hashlib,platform,time
from pathlib import Path
import torch,numpy as np
from .experiment import ROOT,timed
from .scenes import make_scene
from .render import render_dense,Camera
from .tiled import render_tiled,make_plan

def main():
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3);p.add_argument('--repeats',type=int,default=11);a=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('Actual MPS device is required')
    torch.set_num_threads(8);out=ROOT/'results/round05';out.mkdir(parents=True,exist_ok=True);rows=[];start=time.perf_counter()
    for seed in range(a.seeds):
        for resolution,n in [(128,256),(256,1024)]:
            for channels in [3,32,128,512]:
                args=make_scene(seed,n,channels,'mps');cam=Camera(resolution,resolution,resolution*.95);plan=make_plan(*args[:3],cam)
                # Compare the optimized implementation against the same weighted formula.
                with torch.no_grad():ref=render_dense(*args,cam)
                methods=['dense_full','tiled_full','tiled_cached','tiled_cached_half','tiled_cached_channels64','alpha_full']
                random.Random(seed+channels+resolution).shuffle(methods)
                for method in methods:
                    if method=='dense_full':fn=lambda:render_dense(*args,cam)
                    elif method=='alpha_full':fn=lambda:render_dense(*args,cam,mode='alpha')
                    elif method=='tiled_full':fn=lambda:render_tiled(*args,cam)
                    elif method=='tiled_cached_half':fn=lambda:render_tiled(*args,cam,plan=plan,half_accumulation=True)
                    elif method=='tiled_cached_channels64':fn=lambda:render_tiled(*args,cam,plan=plan,channel_batch=64)
                    else:fn=lambda:render_tiled(*args,cam,plan=plan)
                    result,timing=timed(fn,'mps',a.repeats)
                    error=float((result-ref).abs().max());mse=float((result-ref).square().mean())
                    if method!='alpha_full' and error>(.003 if method.endswith('half') else 2e-5):raise RuntimeError(f'Accuracy bound exceeded: {method}: {error}')
                    row={'seed':seed,'resolution':resolution,'gaussians':n,'channels':channels,'method':method,'scope':'cached-plan render' if 'cached' in method else 'full render including plan construction','max_error_weighted_mps':error,'mse_weighted_mps':mse,'candidate_pairs':plan.candidates,'padded_tile_candidates':plan.indices.numel(),'dense_weight_bytes':resolution*resolution*n*4,'padded_tile_weight_bytes':plan.indices.numel()*plan.tile**2*4,'resident_allocated_bytes':torch.mps.current_allocated_memory(),'driver_allocated_bytes':torch.mps.driver_allocated_memory(),'elapsed_seconds':time.perf_counter()-start,**timing};rows.append(row);print(json.dumps(row),flush=True)
                    (out/'raw.json').write_text(json.dumps(rows,indent=2))
                del ref,args,plan,result;torch.mps.empty_cache()
    (out/'manifest.json').write_text(json.dumps({'args':vars(a),'platform':platform.platform(),'torch':torch.__version__,'device':'Apple M4 Pro MPS','elapsed_seconds':time.perf_counter()-start,'timing':'2 warmups + synchronized wall-clock samples; randomized method order per case; 8 CPU threads','memory':'resident allocations are not peak memory measurements; analytical intermediate sizes recorded separately','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
if __name__=='__main__':main()
