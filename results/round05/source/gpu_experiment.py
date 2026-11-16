import json,argparse,platform,hashlib
from pathlib import Path
import torch
from .scenes import make_scene
from .render import render_dense,render_streamed,Camera
from .experiment import timed,ROOT
from .tiled import render_tiled,make_plan

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--round',type=int,default=2);ap.add_argument('--seeds',type=int,default=3);a=ap.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS not available; run with Metal device access')
    torch.set_num_threads(8);out=ROOT/'results'/f'round{a.round:02d}';out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in range(a.seeds):
        for channels in [3,32,128]:
            args=make_scene(seed,256,channels,'mps');cam=Camera(128,128,120)
            with torch.no_grad():ref=render_dense(*(x.cpu().double() for x in args),cam)
            plan=make_plan(*args[:3],cam) if a.round>=3 else None
            for method in (['dense','streamed','alpha'] if a.round<3 else ['dense','tiled_full','tiled_cached','tiled_channels8','alpha']):
                fn=(lambda:render_dense(*args,cam,mode='alpha')) if method=='alpha' else ((lambda:render_streamed(*args,cam)) if method=='streamed' else (lambda:render_dense(*args,cam)))
                if method=='tiled_full':fn=lambda:render_tiled(*args,cam)
                elif method=='tiled_cached':fn=lambda:render_tiled(*args,cam,plan=plan)
                elif method=='tiled_channels8':fn=lambda:render_tiled(*args,cam,plan=plan,channel_batch=8)
                result,timing=timed(fn,'mps',5)
                row={'seed':seed,'channels':channels,'method':method,'max_error_cpu64':float((result.cpu().double()-ref).abs().max()),'allocated_bytes':torch.mps.current_allocated_memory(),**timing};rows.append(row);print(json.dumps(row),flush=True)
    (out/'raw.json').write_text(json.dumps(rows,indent=2));(out/'manifest.json').write_text(json.dumps({'args':vars(a),'device':'mps','torch':torch.__version__,'platform':platform.platform(),'n':256,'image':[128,128],'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}},indent=2))
if __name__=='__main__':main()
