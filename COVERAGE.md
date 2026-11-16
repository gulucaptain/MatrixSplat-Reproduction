# Coverage and scientific limits

| Paper item | Local implementation | Status |
|---|---|---|
| 3D projection and Gaussian response | render.project / alpha_matrix | implemented; 0.3 pixel covariance regularizer is an explicit local convention |
| Eq. 7 normalized weighted feature mixture | blend_reference / blend_matrix | implemented and independently checked |
| Depth attenuation / scene scale / view visibility | weights / train.Appearance | implemented; positive first-order directional visibility is our adaptation |
| Appendix A gradients | autograd + finite differences + analytic quotient tests | checked locally; cutoff boundaries are nondifferentiable |
| Tile-based packing | tiled.make_plan | CPU conservative AABB culling, padded tile batch; not warp-level original |
| Tensor Core MMA | torch.bmm on Apple MPS | different hardware/backend; not reproduced |
| Channel batching | render_tiled channel_batch | supported; can be slower on MPS |
| End-to-end 3DGS optimization | synthetic fixed-geometry training | partial; no densification, pruning or real image dataset |
| Full/cached timings | benchmark.py | scoped explicitly; not identical to original paper protocols |
| Mip-NeRF360 / LERF / 3D-OVS tables | none | not reproduced |
| FlashGS / TC-GS / GEMM-GS / NVIDIA counters | none | requires NVIDIA environment |

## Evidence and assumptions

Paper HTML and hash are saved in sources. No official target implementation was copied. Existing public standard algorithms/frameworks are dependencies. The PLY importer reads DC color and covariance only and cannot preserve higher-order view-dependent color without extension. A generic normalized weighted average can alter occlusion compared with alpha compositing; synthetic retraining demonstrates optimizability, not universal visual equivalence. Both numerical correctness and image quality must be measured.

Timing comparisons against dense all-pairs PyTorch are engineering measurements of this implementation, not comparisons against an optimized production renderer. The largest configured all-pairs weight matrix is 256 MiB (256x256x1024 float32); other intermediate buffers also consume memory. Tile culling reduces the theoretical weight buffer, while driver memory counters include allocator caching.
