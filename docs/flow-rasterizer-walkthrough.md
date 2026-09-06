# Gaussian-Splatting Optical-Flow Rasterizer — End-to-End Walkthrough

**Purpose.** This is a resume/onboarding document. It explains, step by step, how the
differentiable optical-flow rasterizer (`GaussianRasterizerWithFlow` /
`DiffFlowRasterizer`) works — forward **and** backward — and where every piece lives,
so a future session can pick up without re-deriving anything.

Read order for the related docs:
1. **This file** — the full pipeline and where things are.
2. [`flow-forward-pass.md`](flow-forward-pass.md) — forward design rationale.
3. [`flow-backward-design.md`](flow-backward-design.md) — backward derivations (with numbers).

> **Status (2026-06-07):** forward + backward fully implemented. Colour, depth, and
> flow render in one fused pass; colour and flow gradients are complete. Depth has no
> backward. The old cache-based `GaussianFlowRasterizer` path was deleted.
>
> **Cannot be built/tested on the current macOS box** (no `nvcc`, torch-CUDA
> unavailable). Build on a CUDA GPU with `pip install -e .`, then run
> `tests/test_flow_backward.py`.

---

## 0. What it computes

Per output pixel `p`, using **one** set of alpha-compositing weights `w_i = α_i·T_i`
(the *same* weights drive colour, depth, and flow):

```
C(p)    = Σ_i  α_i·T_i · c_i                     (colour, 3×H×W)
depth(p)= z of the Gaussian that drops T below 0.5  (median depth, 1×H×W; default 15)
flow(p) = Σ_i  α_i·T_i · Δᵢ(p)                   (optical flow, 2×H×W, pixel units)
```

`Δᵢ(p) = computePrevPos(p, …) − p` is the per-pixel displacement induced by Gaussian
`i` having moved/scaled/rotated between the **previous** and **current** frame. No
normalisation; background pixels get zero flow. Because flow shares colour's weights,
the backward is tractable in a single extra term — that is the whole design idea.

### Inputs
Current frame: `means3D, scales, rotations` (or `cov3Ds_precomp`), `opacities`,
`sh`/`colors_precomp`. Previous frame: `prev_means3D, prev_scales, prev_rotations`
(or `prev_cov3Ds_precomp`). **No previous-frame opacities or colours** — weights come
from the current frame only.

### Outputs
`forward()` returns `(color [3,H,W], radii [P], depth [1,H,W], flow [2,H,W])`.

---

## 1. Layer stack (call chain)

```
GaussianRasterizerWithFlow.forward()                 # __init__.py:370  (nn.Module)
└ rasterize_gaussians_with_flow()                    # __init__.py:202  (thin wrapper)
  └ _RasterizeGaussiansWithFlow.apply()              # __init__.py:234  (autograd.Function)
    ├ forward:  torch.ops.diff_gaussian_rasterization.rasterize_gaussians_with_flow  # ext.cpp
    │  └ RasterizeGaussiansWithFlowCUDA()            # rasterize_points.cu:237  (tensor bridge)
    │     └ CudaRasterizer::DiffFlowRasterizer::forward()  # rasterizer_impl.cu:545
    │        ├ FORWARD::FLOW::preprocess  → preprocessFlowCUDA   # forward.cu:689 / 550
    │        ├ (CUB sort + tile binning: duplicateWithKeys, identifyTileRanges)
    │        └ FORWARD::DIFF_FLOW::render → diffFlowRenderCUDA    # forward.cu:895 / 769
    └ backward: _C.rasterize_gaussians_with_flow_backward
       └ RasterizeGaussiansWithFlowBackwardCUDA()    # rasterize_points.cu:335
          └ CudaRasterizer::DiffFlowRasterizer::backward()       # rasterizer_impl.cu:705
             ├ BACKWARD::DIFF_FLOW::render    → diffFlowRenderBackwardCUDA  # backward.cu:1081 / 729
             ├ BACKWARD::DIFF_FLOW::preprocess (prev-frame grads)           # backward.cu:1129
             │   ├ addSqrtConicGradCUDA           # backward.cu:932  (current sqrt_conic → conic)
             │   ├ computeCov2DFlowBackwardCUDA   # backward.cu:954  (prev cov2D → prev cov3D + mean)
             │   └ preprocessFlowBackwardCUDA     # backward.cu:1046 (prev mean2D → mean3D, cov3D → scale/rot)
             └ BACKWARD::preprocess (current-frame grads, reused from colour) # backward.cu
```

Files: `diff_gaussian_rasterization/__init__.py` (Python API), `ext.cpp` (custom-op registration),
`rasterize_points.{cu,h}` (torch↔CUDA bridge), `cuda_rasterizer/{rasterizer.h,
rasterizer_impl.{cu,h}, forward.{cu,h}, backward.{cu,h}, auxiliary.h, config.h}`.

---

## 2. Per-Gaussian state buffers

Two opaque byte buffers are allocated by the bridge and carved into typed arrays by
`*State::fromChunk` (see `rasterizer_impl.h`). They are **returned from forward and fed
back into backward** (that's how the backward re-derives intermediate values without
recomputing preprocessing).

`GeometryState` (current frame, shared with the standard rasterizer): `depths`,
`clamped`, `internal_radii`, `means2D`, `cov3D`, `conic_opacity` (`float4` = inverse
cov2D + opacity), `rgb`, `point_offsets`, `tiles_touched`, `bbx_min/max`.

`FlowState` (previous frame + flow extras), `rasterizer_impl.cu:264`:
- `prev_means2D`     — projected previous 2D centre `μ²ᴰ_prev`
- `prev_cov2D`       — **plain** previous 2D covariance (`float3 {a,b,c}`)
- `sqrt_conic`       — matrix sqrt of the **current conic** (inverse cov2D) `√Σ⁻¹_curr`
- `prev_sqrt_cov2D`  — matrix sqrt of the **previous cov2D** `√Σ_prev`
- `prev_cov3D`       — previous 3D covariance (`P×6`); its own buffer (see Gotcha #1)

`ImageState`: `ranges` (per-tile gaussian ranges), `n_contrib` (last contributor per
pixel), `accum_alpha` (final transmittance `T` per pixel — needed by backward).

> **Asymmetry to remember:** the current frame stores the sqrt of the **inverse**
> cov2D (the conic); the previous frame stores the sqrt of the **plain** cov2D. This is
> exactly what "whiten with current, de-whiten with previous" needs (§4).

---

## 3. Forward — preprocessing (`preprocessFlowCUDA`, forward.cu:550)

One thread per Gaussian. Culls by the **current** frustum, then for the survivors:

1. Project current centre → `means2D` (NDC→pixel via `ndc2Pix`). Project previous
   centre (`prev_orig_points`) → `prev_means2D`. Store `depths = p_view.z`.
2. Current cov3D: use `cov3Ds_precomp` if given, else `computeCov3D(scales, rotations)`
   into `geomState.cov3D`. Previous cov3D: `prev_cov3Ds_precomp` if given, else
   `computeCov3D(prev_scales, prev_rotations)` into **`flowState.prev_cov3D`**.
3. Current cov2D = `computeCov2D(...)` (EWA projection + 0.3 low-pass on the diagonal),
   invert → `conic`; pack `conic_opacity = {conic, opacity}`. Previous cov2D =
   `computeCov2D(prev …)`; store the plain `prev_cov2D` (no inversion).
4. `sqrt_conic = computeSquareRootCov2D(conic)` and
   `prev_sqrt_cov2D = computeSquareRootCov2D(prev_cov)` (forward.cu:683–684).
5. SH→RGB into `rgb` (if not precomputed). Compute screen-space radius / tile bbox →
   `tiles_touched` for binning.

`computeSquareRootCov2D` (forward.cu:507) does this via eigendecomposition, but for a
PSD symmetric 2×2 it equals the closed form `√M = (M + √det·I)/√(tr+2√det)` — that
identity is what makes the backward clean (§6).

After preprocessing: CUB prefix-sum on `tiles_touched`, `duplicateWithKeys` +
radix-sort by (tile, depth), `identifyTileRanges` → `imgState.ranges`.

---

## 4. Forward — render (`diffFlowRenderCUDA`, forward.cu:769)

Tile-based, one thread per pixel, one `BLOCK_X×BLOCK_Y` tile per block. Gaussians are
streamed front-to-back in shared-memory batches (`collected_*`). For each Gaussian that
passes `power ≤ 0`, `alpha ≥ 1/255`, and `test_T = T·(1−alpha) ≥ 1e-4`:

```
alpha = min(0.99, opacity · exp(power));   w = alpha · T
C[ch] += rgb[ch] · w
if (T > 0.5 && test_T < 0.5)  depth = this Gaussian's z      // median depth
Δ = computePrevPos(p, prev_means2D, means2D, sqrt_conic, prev_sqrt_cov2D) − p
flow_x += w · Δ.x ;  flow_y += w · Δ.y
T = test_T
```

After the loop: `accum_alpha[pix]=T`, `n_contrib[pix]=last_contributor`,
`out_color = C + T·bg`, `out_depth = depth`, `out_flow = {flow_x, flow_y}`.

### `computePrevPos` (forward.cu:934) — the displacement model

With `S = √conic_curr`, `P = √cov2D_prev`, both symmetric 2×2 as `float3 {a,b,c}`:

```
diff     = p − μ²ᴰ_curr
norm_pos = S · diff                       # whiten into current Gaussian's unit frame
prev_pos = P · norm_pos + μ²ᴰ_prev        # de-whiten into previous Gaussian's frame
Δ        = prev_pos − p
```

Intuition: carry pixel `p` into the canonical frame of the current Gaussian, then push
it out through the previous Gaussian's shape/position. A Gaussian that translated,
rotated, or scaled between frames yields a per-pixel displacement field, not just a
rigid centre shift.

---

## 5. Backward — overview

The combined render backward replays the tile loop **back-to-front**, reconstructing
`T` via `T = T/(1−alpha)`, and accumulates gradients with `atomicAdd`. Colour gradients
are identical to the standard `renderCUDA`. Flow adds two things per Gaussian:

**(a) Through the weight α** — treat `Δᵢ` like an extra colour channel:
```
dL/dα_i += (Δᵢ.x − accum_rec_flowx)·dL/dflow_x + (Δᵢ.y − accum_rec_flowy)·dL/dflow_y
```
folded into the same `dL_dalpha` as colour, so the existing `α→{conic, mean2D, opacity}`
block handles it unchanged.

**(b) Through the displacement Δ directly** — `dL/dΔᵢ = (α_i·T_i)·dL/dflow`, routed by
`computePrevPos`'s backward into `prev_means2D`, `prev_sqrt_cov2D`, `sqrt_conic`, and
(current) `means2D`. These need three new per-Gaussian scratch buffers:
`dL_dsqrt_conic`, `dL_dprev_mean2D`, `dL_dprev_sqrt_cov2D` (cudaMalloc'd inside
`DiffFlowRasterizer::backward`, rasterizer_impl.cu:705).

Then a preprocess stage maps the 2D gradients to 3D parameters:

| 2D gradient | → mapped by | → into |
|---|---|---|
| `dL_dmean2D` (colour+flow) | `BACKWARD::preprocess` (reused) | `means3D` |
| `dL_dsqrt_conic` | `addSqrtConicGradCUDA` → folds into `dL_dconic`, then `computeCov2DCUDA` | `cov3D`, `means3D` → `scales`, `rotations` |
| `dL_dprev_sqrt_cov2D` | `computeCov2DFlowBackwardCUDA` | `prev_cov3D`, `prev_means3D` |
| `dL_dprev_mean2D` | `preprocessFlowBackwardCUDA` (projection) | `prev_means3D` |
| `dL_dprev_cov3D` | `computeCov3D` backward (in `preprocessFlowBackwardCUDA`) | `prev_scales`, `prev_rotations` |

**Order matters:** `BACKWARD::DIFF_FLOW::preprocess` runs `addSqrtConicGradCUDA`
*before* `BACKWARD::preprocess` (so the folded `dL_dconic` is seen by `computeCov2DCUDA`);
and `computeCov2DFlowBackwardCUDA` (which **assigns** `dL_dprev_mean3D`) runs before
`preprocessFlowBackwardCUDA` (which **+=** the projection part).

---

## 6. Backward — the two derivations (both numerically verified, ~1e-9)

### `computeSquareRootCov2DBackward` (backward.cu:672)
Differentiates the closed form `√M=(M+√det·I)/√(tr+2√det)` analytically — no
eigendecomposition. With `s=√det`, `t=√(tr+2s)`, outputs `X=(a+s)/t, Y=b/t, Z=(c+s)/t`,
and `dO/dν = (dN/dν − O·dt/dν)/t`. Used **twice**: on the current `conic` (in
`addSqrtConicGradCUDA`, result added to colour's `dL_dconic`) and on the previous
`prev_cov2D` (in `computeCov2DFlowBackwardCUDA`). Full formulas: backward-design doc §2.2.

### `computePrevPos` backward (inlined in `diffFlowRenderBackwardCUDA`, backward.cu:729)
With `q = (α·T)·dL/dflow`, `norm_pos`, `diff = p−μ²ᴰ_curr`:
```
dL/dμ²ᴰ_prev += q
dL/dP        += (q.x·np.x,  q.x·np.y+q.y·np.x,  q.y·np.y)          # P = √cov2D_prev
h = (q.x·P.a+q.y·P.b,  q.x·P.b+q.y·P.c)
dL/dS        += (h.x·diff.x, h.x·diff.y+h.y·diff.x, h.y·diff.y)    # S = √conic_curr
dL/dμ²ᴰ_curr += −(h.x·S.a+h.y·S.b,  h.x·S.b+h.y·S.c)
```
The off-diagonal `b` is one stored parameter (appears in two matrix slots), so there is
no factor-of-two here; the symmetric factor-of-two lives later in `cov2D→cov3D`.

### Previous-frame cov2D→cov3D (`computeCov2DFlowBackwardCUDA`, backward.cu:954)
A clone of the colour `computeCov2DCUDA` that **skips the conic-inverse Jacobian**
(prev stores the plain cov2D) and uses `prev_means3D` + `prev_cov3D`. It first calls
`computeSquareRootCov2DBackward(prev_cov2D)` then the `T=W·J`, `cov2D=Tᵀ·Vrkᵀ·T` mapping.

---

## 7. Gradients returned

C++ `RasterizeGaussiansWithFlowBackwardCUDA` returns 12 tensors:
`dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales,
dL_drotations, dL_dprev_means3D, dL_dprev_cov3D, dL_dprev_scales, dL_dprev_rotations`.

Python `_RasterizeGaussiansWithFlow.backward` (__init__.py:293) reorders these into the
13-slot grad tuple matching the forward inputs
`(means3D, prev_means3D, means2D, sh, colors_precomp, opacities, scales, prev_scales,
rotations, prev_rotations, cov3Ds_precomp, prev_cov3Ds_precomp, raster_settings=None)`.

The forward returns a 9-tuple that includes **`flowBuffer`** (the saved `FlowState`),
which is stashed in `ctx` and handed back to the backward.

---

## 8. Gotchas / invariants (read before editing)

1. **`prev_cov3D` must stay in its own buffer.** The original forward wrote prev cov3D
   over `geomState.cov3D`; that was fixed by adding `FlowState::prev_cov3D`. The
   current-frame backward (`computeCov2DCUDA`) reads `geomState.cov3D` expecting the
   *current* matrix — don't reintroduce the overwrite.
2. **`prev_means2D` must project from `prev_orig_points`.** A fixed bug had it using
   `orig_points`, making `prev_means2D == means2D` (zero-flow passed trivially,
   translation could not). See `preprocessFlowCUDA`.
3. **Empty precomp tensors → null pointers.** The `cov3D_precomp != nullptr` style
   checks rely on PyTorch returning a null `data_ptr()` for size-0 tensors; the prev
   path follows the same convention (`prev_cov3D_ptr` in `DiffFlowRasterizer::backward`).
4. **No previous-frame opacity/colour.** Weights are current-frame only. (The unused
   `prev_opacities` placeholder and the opacity slot of the old `prev_cov2D_opacity`
   `float4` were removed; `prev_cov2D` is now a clean `float3`.)
5. **Depth has no backward.** `grad_depth` is ignored (default-depth fallback makes it
   non-differentiable), consistent with the standard rasterizer.
6. **gradcheck caveat.** Kernels are float32, so use finite differences with a loose
   tolerance (see `tests/test_flow_backward.py`), not `torch.autograd.gradcheck`
   (which assumes float64).
7. **Cache path is gone.** `GaussianFlowRasterizer`, `create_cache`, `FlowRasterizer`,
   `FORWARD::FLOW::render`, the cache/header state structs and kernels were deleted.
   `FORWARD::FLOW::preprocess` is kept (shared by DIFF_FLOW). Don't look for them.

---

## 9. Build, test, verify

```bash
pip install -e .                      # needs a CUDA GPU + matching torch/CUDA
python tests/test_flow_backward.py    # zero-flow, uniform-translation, finite-diff grad check
```

Sanity checks the tests encode (from the forward-pass doc):
- **Zero-flow:** `prev_* == current_*` ⇒ `flow ≡ 0` and all `dL/dprev_* ≡ 0`.
- **Uniform translation:** shift `prev_means3D` ⇒ foreground flow ≈ Δ.
- **Finite difference:** analytic vs numerical grad on `means3D` and `prev_means3D`.

If something looks off in the backward, the two highest-risk pieces are
`computeSquareRootCov2DBackward` and the `computePrevPos` routing — both have standalone
NumPy validations reproduced in `flow-backward-design.md` (and were checked to ~1e-9).

---

## 10. Possible next steps (not yet done)

- Depth backward (currently a hard non-differentiable median fallback).
- Optionally drop the now-unused `bbx_min/bbx_max` from `GeometryState` (only the
  removed cache layout used them).
- A real end-to-end gradcheck harness with a known camera, run on GPU.
