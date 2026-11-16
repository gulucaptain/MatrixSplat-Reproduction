import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .experiment import ROOT

def main():
    out=ROOT/'results/round05';rows=json.loads((out/'raw.json').read_text());training=json.loads((ROOT/'results/round04/raw.json').read_text())
    methods=['dense_full','tiled_full','tiled_cached','tiled_cached_half','tiled_cached_channels64','alpha_full'];colors=plt.cm.tab10.colors
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained');summary=[]
    for ax,resolution in zip(axes,[128,256]):
        for j,method in enumerate(methods):
            means=[];lows=[];highs=[]
            for channels in [3,32,128,512]:
                group=[r for r in rows if r['resolution']==resolution and r['channels']==channels and r['method']==method]
                x=np.array([r['median_ms'] for r in group]);means.append(np.median(x));lows.append(x.min());highs.append(x.max())
                summary.append({'resolution':resolution,'channels':channels,'method':method,'median_of_seed_medians_ms':float(np.median(x)),'min_seed_median_ms':float(x.min()),'max_seed_median_ms':float(x.max()),'max_error':max(r['max_error_weighted_mps'] for r in group)})
            ax.plot(range(4),means,'o-',color=colors[j],label=method);ax.fill_between(range(4),lows,highs,color=colors[j],alpha=.1)
        ax.set_xticks(range(4),[3,32,128,512]);ax.set(xlabel='Feature dimensions',ylabel='Milliseconds (log)',yscale='log',title=f'{resolution} x {resolution}; seed median range');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'timing.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for r in training:axes[0].plot([x['step'] for x in r['history']],[x['loss'] for x in r['history']],label=f"seed {r['seed']}")
    axes[0].set(yscale='log',xlabel='Optimizer steps',ylabel='MSE',title='Synthetic multiview training');axes[0].legend()
    x=np.arange(len(training));axes[1].bar(x-.17,[r['heldout_psnr_before'] for r in training],.34,label='Before');axes[1].bar(x+.17,[r['heldout_psnr_after'] for r in training],.34,label='After');axes[1].set(xlabel='Seed',ylabel='RGB PSNR (dB)',title='Held-out synthetic view');axes[1].legend();fig.savefig(out/'training.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,4));ax.imshow(Image.open(ROOT/'results/round04/seed00_heldout.png'));ax.axis('off');ax.set_title('Synthetic alpha target                 Weighted initialization                 Learned weighted model');fig.tight_layout();fig.savefig(out/'render_comparison.png',dpi=170);plt.close(fig)
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    lines=['# Local Apple GPU results','',f'{len(rows)} benchmark configurations, three scene seeds, 11 synchronized repetitions per configuration. All images/scenes are synthetic.','', '| Resolution | Features | Dense weighted full (ms) | Tiled full (ms) | Local ratio |','|---|---:|---:|---:|---:|']
    for res in [128,256]:
        for c in [3,32,128,512]:
            get=lambda m:next(r['median_of_seed_medians_ms'] for r in summary if r['resolution']==res and r['channels']==c and r['method']==m)
            d,t=get('dense_full'),get('tiled_full');lines.append(f'| {res} | {c} | {d:.3f} | {t:.3f} | {d/t:.2f}x |')
    full=max(r['max_error_weighted_mps'] for r in rows if r['method'] in ['tiled_full','tiled_cached']);half=max(r['max_error_weighted_mps'] for r in rows if r['method']=='tiled_cached_half')
    lines+=['',f'Maximum FP32 tiled-versus-dense weighted difference: {full:.3g}. Half-matmul difference: {half:.3g}.',f"Held-out RGB PSNR averaged across five synthetic training seeds: {np.mean([r['heldout_psnr_before'] for r in training]):.2f} -> {np.mean([r['heldout_psnr_after'] for r in training]):.2f} dB.",'','## Interpretation','Ratios above compare our dense and tiled mathematical implementations on the same Apple GPU. They are NOT speedups over optimized official CUDA 3DGS, NOT NeurIPS table reproductions, and NOT evidence of NVIDIA Tensor Core execution. Dense all-pairs rasterization intentionally serves as a correctness-oriented baseline; most tiled gains can come from culling.','Cached-plan timing excludes CPU candidate construction and must not be compared as if it had identical scope to full rendering. The alpha renderer uses a different compositing model, so its output difference is a model difference rather than an implementation error.','Timing variability is shown as the range of seed medians; raw samples remain available. Resident memory counters are not peaks. Half precision may be slower, and fixed narrow channel batches may add dispatch overhead.','', 'Figures: [timing](timing.png), [training](training.png), [render comparison](render_comparison.png).']
    (out/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines))
if __name__=='__main__':main()
