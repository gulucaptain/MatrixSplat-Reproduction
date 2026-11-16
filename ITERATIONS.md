# Five development cycles

1. **Mathematical reference:** perspective covariance projection, ordered alpha compositing, scalar weighted accumulation and dense matrix equivalent. CPU experiment on three seeds; finite-difference checks. Establish that alpha-versus-weighted differences are model differences, not numerical bugs.
2. **Actual Apple GPU execution:** verify MPS availability by running a matrix operation outside the restricted process; run three seeds at 3/32/128 channels. Add bounded pixel streaming and channel slicing. Record that added dispatches can make streaming slower.
3. **Spatial optimization:** add conservative tile culling, padded candidate packing and batched matrix multiplication. Test non-multiple image dimensions, empty tiles and derivative equivalence. Measure full versus cached-plan timing separately; narrow 8-channel slices are not uniformly beneficial.
4. **Optimization and input:** add standard 3DGS PLY importer and five-seed multiview synthetic training, 240 Adam steps each. Learn RGB/semantic features, opacity, directional visibility, scale and background with fixed geometry. Verify full geometry gradients against dense reference. Save checkpoints and held-out-view images.
5. **Scale/precision audit:** add half-precision matrix operands; expand to 128/256 resolution, 256/1024 Gaussians, 3/32/128/512 features, three seeds and 11 synchronized repeated samples. Randomize method order, preserve all failures/slowdowns, report timing scopes, compare analytical derivatives and permutation invariance, and generate final figures/report.

Source snapshots and manifests accompany all five cycles. Completing these cycles does not make the original paper's hardware results reproduced.
