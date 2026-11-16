import argparse,time,json,platform,hashlib
from pathlib import Path
import torch,numpy as np
from .render import Camera,render_dense
from .scenes import make_scene

ROOT=Path(__file__).resolve().parents[2]
def synchronize(device):
    if device=='mps':torch.mps.synchronize()

def timed(fn,device,repeats=5):
    with torch.no_grad():
        for _ in range(2):fn()
        synchronize(device);times=[]
        for _ in range(repeats):
            synchronize(device);start=time.perf_counter();result=fn();synchronize(device);times.append(time.perf_counter()-start)
    return result,{'median_ms':float(np.median(times)*1000),'min_ms':min(times)*1000,'samples_ms':[x*1000 for x in times]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,default=1);ap.add_argument('--device',default='cpu');ap.add_argument('--seeds',type=int,default=3);a=ap.parse_args();torch.set_num_threads(8)
    if a.device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no silent CPU fallback')
    out=ROOT/'results'/f'round{a.round:02d}';out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in range(a.seeds):
        args=make_scene(seed,128,3,a.device);cam=Camera()
        ref=render_dense(*args,cam,mode='reference')
        for mode in ['reference','matrix','alpha']:
            result,timing=timed(lambda:render_dense(*args,cam,mode=mode),a.device)
            row={'seed':seed,'mode':mode,'device':a.device,'max_error_to_weighted_reference':float((result-ref).abs().max()),**timing};rows.append(row);print(json.dumps(row),flush=True)
    (out/'raw.json').write_text(json.dumps(rows,indent=2));(out/'manifest.json').write_text(json.dumps({'args':vars(a),'torch':torch.__version__,'platform':platform.platform(),'mps':torch.backends.mps.is_available(),'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
if __name__=='__main__':main()
