# Differentiable Optical Flow Forward Pass

This document describes the design and implementation of the differentiable optical flow renderer added to this repository (`GaussianRasterizerWithFlow` / `_C.rasterize_gaussians_with_flow`).

## Motivation

A previous **cache-based** flow rasterizer (`GaussianFlowRasterizer`, since removed)
rendered optical flow but was **not differentiable**:
- It weighted contributions by `T_prev · G_prev · T_curr · G_curr`, combining transmittances from two separate forward passes.
- It normalised the result by `Σ weights`, which has no backward implementation.
- Its backward silently ignored all flow gradients.

This implementation uses **identical compositing weights for colour and flow**,
making the backward tractable with no extra forward pass or cache required. It has
fully replaced the cache-based path.

## Mathematical Formulation

During standard alpha-compositing each pixel `p` accumulates:

```
C(p)    = Σ_i  α_i(p) · T_i(p) · c_i           (colour)
flow(p) = Σ_i  α_i(p) · T_i(p) · Δᵢ(p)         (optical flow)
```

where `w_i = α_i · T_i` are the **same** colour compositing weights, and `Δᵢ(p)` is the per-pixel displacement for Gaussian `i`.

### Displacement via Gaussian whitening

```
Δᵢ(p) = computePrevPos(p, μ²ᴰ_prev,i, μ²ᴰ_curr,i, √Σ²ᴰ_curr,i, √Σ²ᴰ_prev,i) − p
```

`computePrevPos` maps pixel `p` into the normalised space of the current Gaussian (whitening with `√Σ²ᴰ_curr`), then de-whitens with the previous Gaussian's covariance (`√Σ²ᴰ_prev`) and shifts by `μ²ᴰ_prev`.  This device function is defined in `cuda_rasterizer/forward.cu`.

### Key differences from the removed cache-based `GaussianFlowRasterizer`

| Aspect | cache-based `GaussianFlowRasterizer` (removed) | `GaussianRasterizerWithFlow` |
|--------|--------------------------|------------------------------|
| Weights | `T_prev·G_prev·T_curr·G_curr` | `α_i·T_i` (identical to colour) |
| Normalisation | divides by `Σ weights` | none |
| Cache required | yes | no |
| Backward | not implemented | colour **and** flow gradients implemented |

Dropping normalisation is intentional: `Σ_i α_i T_i ≤ 1` naturally, and background pixels receive zero flow (consistent with how background colour is handled).

## Required Inputs

Both current and previous frame parameters are required. Previous-frame opacities are **not** needed (weights come from the current frame only).

| Parameter | Current frame | Previous frame |
|-----------|--------------|----------------|
| 3D positions | `means3D` | `prev_means3D` |
| Scales | `scales` | `prev_scales` |
| Rotations | `rotations` | `prev_rotations` |
| Precomputed cov3D | `cov3Ds_precomp` | `prev_cov3Ds_precomp` |
| Opacities | `opacities` | — |
| SHs / colours | `sh` / `colors_precomp` | — |

## Python API

```python
from diff_gaussian_rasterization import GaussianRasterizerWithFlow, GaussianRasterizationSettings

raster_settings = GaussianRasterizationSettings(...)

rasterizer = GaussianRasterizerWithFlow(raster_settings)

color, radii, depth, flow = rasterizer.forward(
    means3D=means3D,           # [P, 3]
    means2D=means2D,           # [P, 3]  (screenspace, for grad accumulation)
    opacities=opacities,       # [P, 1]
    prev_means3D=prev_means3D, # [P, 3]
    shs=shs,                   # [P, K, 3]  (or use colors_precomp)
    scales=scales,             # [P, 3]
    prev_scales=prev_scales,   # [P, 3]
    rotations=rotations,       # [P, 4]
    prev_rotations=prev_rotations,  # [P, 4]
)
# color : [3, H, W]
# radii : [P]
# depth : [1, H, W]
# flow  : [2, H, W]  — (flow_x, flow_y) in pixel units
```

The lower-level function is also available:

```python
from diff_gaussian_rasterization import rasterize_gaussians_with_flow
color, radii, depth, flow = rasterize_gaussians_with_flow(
    means3D, prev_means3D, means2D, sh, colors_precomp,
    opacities, scales, prev_scales, rotations, prev_rotations,
    cov3Ds_precomp, prev_cov3Ds_precomp, raster_settings)
```

## Data Flow

```
GaussianRasterizerWithFlow.forward()
  → rasterize_gaussians_with_flow()          # Python
  → _RasterizeGaussiansWithFlow.apply()      # torch.autograd.Function
  → torch.ops.diff_gaussian_rasterization.rasterize_gaussians_with_flow()  # custom op (ext.cpp)
  → RasterizeGaussiansWithFlowCUDA()         # rasterize_points.cu
  → CudaRasterizer::DiffFlowRasterizer::forward()  # rasterizer_impl.cu
  → FORWARD::FLOW::preprocess kernel         # forward.cu  (reused)
  → FORWARD::DIFF_FLOW::render kernel        # forward.cu  (diffFlowRenderCUDA)
```

## Implementation Details

### Preprocessing (`FORWARD::FLOW::preprocess`, `preprocessFlowCUDA`)

Reused from the existing flow rasterizer.  For each Gaussian it computes and stores:

- `means2D` — projected 2D centre (current frame)
- `prev_means2D` — projected 2D centre (previous frame)
- `conic_opacity` — inverse 2D covariance + opacity (current frame)
- `sqrt_conic` — matrix square root of inverse 2D covariance (current frame)
- `prev_sqrt_cov2D` — matrix square root of 2D covariance (previous frame)

Both square-root matrices are symmetric 2×2, stored as `float3 {a, b, c}`.

### Render kernel (`diffFlowRenderCUDA`)

Per-pixel accumulation, identical to the standard colour kernel except for two additional accumulators:

```cuda
// Shared memory additions (loaded alongside collected_xy):
__shared__ float2 collected_prev_xy[BLOCK_SIZE];
__shared__ float3 collected_sqrt_conic[BLOCK_SIZE];
__shared__ float3 collected_prev_sqrt_cov2D[BLOCK_SIZE];

// Inside the per-Gaussian loop (after computing alpha, T — same as colour):
float2 prev_pixf_j = computePrevPos(
    pixf,
    collected_prev_xy[j],
    collected_xy[j],
    collected_sqrt_conic[j],
    collected_prev_sqrt_cov2D[j]);

flow_x += alpha * T * (prev_pixf_j.x - pixf.x);
flow_y += alpha * T * (prev_pixf_j.y - pixf.y);

// Written out after the loop:
out_flow[pix_id]         = flow_x;
out_flow[H * W + pix_id] = flow_y;
```

## Backward Pass

The backward pass is **implemented** (`_C.rasterize_gaussians_with_flow_backward` →
`CudaRasterizer::DiffFlowRasterizer::backward`). The autograd backward
(`_RasterizeGaussiansWithFlow.backward`) now returns real gradients for **all**
inputs: current-frame `means3D`, `means2D`, `scales`, `rotations`, `opacities`,
`cov3Ds_precomp`, `sh`, `colors_precomp`, **and** previous-frame `prev_means3D`,
`prev_scales`, `prev_rotations`, `prev_cov3Ds_precomp`.

Because flow uses the same `α_i · T_i` weights as colour, the backward kernel
(`diffFlowRenderBackwardCUDA`) follows the colour `renderCUDA` structure with one
extra term per pixel:

```
dL/d(α_i) += dL/d(flow_x) · T_i · Δᵢ.x + dL/d(flow_y) · T_i · Δᵢ.y
```

plus a gradient through the displacement `Δᵢ` itself (`dL/dΔᵢ = α_i·T_i·dL/dflow`),
which `computePrevPos`'s backward routes to:
- `means2D_curr` and `sqrt_conic_curr` → into `scales`, `rotations`, `means3D`
- `means2D_prev` and `prev_sqrt_cov2D` → into `prev_scales`, `prev_rotations`, `prev_means3D`

The matrix-square-root backward uses the closed form for symmetric 2×2 PSD matrices.
The full derivation and the per-layer wiring are documented in
[`flow-backward-design.md`](flow-backward-design.md).

> Depth has no backward (default-depth fallback makes it non-differentiable); a
> `grad_depth` passed in is ignored, consistent with the standard rasterizer.

## Verification

```python
import torch
from diff_gaussian_rasterization import GaussianRasterizerWithFlow, GaussianRasterizationSettings

# 1. Zero-flow test:
#    prev_means3D = means3D, prev_scales = scales, prev_rotations = rotations
#    → flow should be exactly zero everywhere

# 2. Uniform translation test:
#    translate all prev_means3D by constant pixel offset Δ
#    → all foreground pixels should have flow ≈ Δ

# 3. Gradient check (colour grads only):
torch.autograd.gradcheck(rasterize_gaussians_with_flow, inputs, eps=1e-3)
```