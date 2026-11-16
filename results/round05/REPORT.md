# Local Apple GPU results

144 benchmark configurations, three scene seeds, 11 synchronized repetitions per configuration. All images/scenes are synthetic.

| Resolution | Features | Dense weighted full (ms) | Tiled full (ms) | Local ratio |
|---|---:|---:|---:|---:|
| 128 | 3 | 48.193 | 12.245 | 3.94x |
| 128 | 32 | 51.081 | 10.479 | 4.87x |
| 128 | 128 | 54.977 | 14.121 | 3.89x |
| 128 | 512 | 55.172 | 19.022 | 2.90x |
| 256 | 3 | 769.814 | 37.636 | 20.45x |
| 256 | 32 | 790.314 | 35.760 | 22.10x |
| 256 | 128 | 791.158 | 41.738 | 18.96x |
| 256 | 512 | 815.444 | 64.965 | 12.55x |

Maximum FP32 tiled-versus-dense weighted difference: 3.58e-07. Half-matmul difference: 0.00107.
Held-out RGB PSNR averaged across five synthetic training seeds: 10.66 -> 33.21 dB.

## Interpretation
Ratios above compare our dense and tiled mathematical implementations on the same Apple GPU. They are NOT speedups over optimized official CUDA 3DGS, NOT NeurIPS table reproductions, and NOT evidence of NVIDIA Tensor Core execution. Dense all-pairs rasterization intentionally serves as a correctness-oriented baseline; most tiled gains can come from culling.
Cached-plan timing excludes CPU candidate construction and must not be compared as if it had identical scope to full rendering. The alpha renderer uses a different compositing model, so its output difference is a model difference rather than an implementation error.
Timing variability is shown as the range of seed medians; raw samples remain available. Resident memory counters are not peaks. Half precision may be slower, and fixed narrow channel batches may add dispatch overhead.

Figures: [timing](timing.png), [training](training.png), [render comparison](render_comparison.png).