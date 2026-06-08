# Differentiable Optical Flow — Backward Pass Findings & Design

> **Status: implemented.** This document records the analysis and the analytic
> derivation; the design below has been built. Code: `diffFlowRenderBackwardCUDA`,
> `computeSquareRootCov2DBackward`, `computeCov2DFlowBackwardCUDA`,
> `preprocessFlowBackwardCUDA`, `addSqrtConicGradCUDA` and the
> `BACKWARD::DIFF_FLOW::{render,preprocess}` wrappers in `cuda_rasterizer/backward.cu`;
> orchestration in `CudaRasterizer::DiffFlowRasterizer::backward`
> (`rasterizer_impl.cu`); bridge `RasterizeGaussiansWithFlowBackwardCUDA`; Python
> `_RasterizeGaussiansWithFlow.backward`. The two forward bugs in §3 were fixed and
> the previous cov3D now lives in its own `FlowState::prev_cov3D` buffer.

This document records the findings from a deep read of the differentiable optical
flow path (`GaussianRasterizerWithFlow` / `DiffFlowRasterizer`) and lays out the
analytic backward derivation required to make the flow-specific gradients
(`prev_means3D`, `prev_scales`, `prev_rotations`, `prev_cov3Ds_precomp`, plus flow
contributions to the current-frame parameters) work.

It complements [`flow-forward-pass.md`](flow-forward-pass.md), which describes the
forward design. Read that first.

---

## 1. Forward path, as actually implemented

The differentiable flow forward is a single fused kernel that produces colour,
depth **and** flow with the *same* `α·T` compositing weights.

```
DiffFlowRasterizer::forward                     # rasterizer_impl.cu
  → FORWARD::FLOW::preprocess  (preprocessFlowCUDA)   # forward.cu:1142
  → FORWARD::DIFF_FLOW::render (diffFlowRenderCUDA)    # forward.cu:1360
```

### 1.1 Per-Gaussian preprocessing — `preprocessFlowCUDA` (forward.cu:1142)

For every Gaussian it stores, in addition to the usual geometry state:

| Buffer | Meaning | Source line |
|--------|---------|-------------|
| `points_xy_image[idx]`     | current 2D centre `μ²ᴰ_curr` | forward.cu:1271 |
| `prev_points_xy_image[idx]`| previous 2D centre `μ²ᴰ_prev` | forward.cu:1272 |
| `conic_opacity[idx]`       | inverse current cov2D + opacity | forward.cu:1274 |
| `prev_cov2D[idx]`          | **non-inverted** previous cov2D (`float3`) | `preprocessFlowCUDA` |
| `sqrt_conic[idx]`          | matrix sqrt of the current **conic** (inverse cov2D) | forward.cu:1276 |
| `prev_sqrt_cov2D[idx]`     | matrix sqrt of the previous cov2D | forward.cu:1277 |

Note the asymmetry that matters for the backward: the current frame stores the
square root of the **inverse** covariance (`conic`), the previous frame stores the
square root of the **plain** covariance. This is exactly what whitening then
de-whitening needs (see §1.3).

### 1.2 Render kernel — `diffFlowRenderCUDA` (forward.cu:1360)

Identical front-to-back alpha compositing as colour. The flow-specific additions
(forward.cu:1460-1469):

```
float2 prev_pixf_j = computePrevPos(pixf,
                                    collected_prev_xy[j],   // μ²ᴰ_prev
                                    collected_xy[j],        // μ²ᴰ_curr
                                    collected_sqrt_conic[j],     // √conic_curr
                                    collected_prev_sqrt_cov2D[j]);// √cov2D_prev
flow_x += alpha * T * (prev_pixf_j.x - pixf.x);
flow_y += alpha * T * (prev_pixf_j.y - pixf.y);
```

So with `Δᵢ(p) = computePrevPos(...) − p` and `wᵢ = αᵢ·Tᵢ`:

```
flow(p) = Σ_i  wᵢ · Δᵢ(p)
```

— the same `wᵢ` as colour, no normalisation. Background pixels get zero flow.

### 1.3 `computePrevPos` (forward.cu:1527) — the displacement model

```
diff      = p − μ²ᴰ_curr
norm_pos  = √conic_curr · diff          # whiten into the unit Gaussian
prev_pos  = √cov2D_prev · norm_pos + μ²ᴰ_prev   # de-whiten into the prev Gaussian
```

Both `√conic_curr` (`S`) and `√cov2D_prev` (`P`) are symmetric 2×2 matrices stored
as `float3 {a, b, c}` = `[[a, b], [b, c]]`. Written out:

```
norm_pos.x = S.a·diff.x + S.b·diff.y
norm_pos.y = S.b·diff.x + S.c·diff.y
prev_pos.x = P.a·norm_pos.x + P.b·norm_pos.y + μ²ᴰ_prev.x
prev_pos.y = P.b·norm_pos.x + P.c·norm_pos.y + μ²ᴰ_prev.y
```

Intuition: map the pixel into the canonical (unit-variance) frame of the current
Gaussian, then push it out into the previous Gaussian's frame. A Gaussian that
moved/rotated/scaled between frames produces a per-pixel displacement field, not
just a rigid centre shift.

### 1.4 `computeSquareRootCov2D` (forward.cu:1100)

The forward computes the symmetric matrix square root via an explicit
eigendecomposition (trace/discriminant → eigenvalues → eigenvectors via
`atan2`). For a PSD symmetric 2×2 matrix this is **analytically equal** to the
much friendlier closed form

```
s = √det(M)                         # det = a·c − b²
t = √(trace(M) + 2s)                # trace = a + c
√M = (M + s·I) / t
    ⇒  X = (a + s)/t,  Y = b/t,  Z = (c + s)/t
```

This identity is the key that makes the backward tractable — see §2.2. (The
forward's eps-clamps are ignored in the backward, standard practice.)

---

## 2. Backward derivation

Per the forward, flow shares the colour weights, so the render backward mirrors the
existing `renderCUDA` (backward.cu:399) with flow treated as **two extra channels
whose per-Gaussian "feature" is `Δᵢ(p)` itself** — except `Δᵢ` is not a constant
feature, it depends on the Gaussian's 2D parameters, so it needs a second route.

### 2.1 Render backward — two contributions per Gaussian

Let `g_fx = dL/d(flow_x)[p]`, `g_fy = dL/d(flow_y)[p]` be the incoming pixel
gradients, and `w = αᵢ·Tᵢ`.

**(a) Through the weight `α` (same machinery as colour).** Treat `Δᵢ` like a
colour channel value. Maintain back-to-front running accumulators
`accum_rec_flow{x,y}` exactly as colour's `accum_rec[ch]`:

```
dL_dalpha += (Δᵢ.x − accum_rec_flowx) · g_fx
           + (Δᵢ.y − accum_rec_flowy) · g_fy
```

This term is **added into the same `dL_dalpha`** as colour, then the existing
`α → G → {conic, mean2D, opacity}` block (backward.cu:537-554) propagates it with no
change. No new buffers needed for this part.

**(b) Through the displacement `Δᵢ` directly.** Since `flow = Σ wᵢ Δᵢ` and `p` is a
constant pixel, `dL/dΔᵢ = wᵢ · (g_fx, g_fy)`. Call this `(qx, qy) = (w·g_fx, w·g_fy)`
and back-propagate through `computePrevPos` (`Δ = prev_pos − p`, so `dΔ = dprev_pos`).

Using the forward equations of §1.3 with `S = √conic_curr`, `P = √cov2D_prev`,
`diff = p − μ²ᴰ_curr`, `norm_pos` as defined:

```
# previous-frame centre
dL/dμ²ᴰ_prev   += (qx, qy)

# √cov2D_prev  (P)
dL/dP.a += qx·norm_pos.x
dL/dP.b += qx·norm_pos.y + qy·norm_pos.x
dL/dP.c += qy·norm_pos.y

# back through P into norm_pos
hx = qx·P.a + qy·P.b
hy = qx·P.b + qy·P.c

# √conic_curr (S)
dL/dS.a += hx·diff.x
dL/dS.b += hx·diff.y + hy·diff.x
dL/dS.c += hy·diff.y

# back through S into diff = p − μ²ᴰ_curr   (so dμ²ᴰ_curr = −d diff)
dL/dμ²ᴰ_curr.x += −(hx·S.a + hy·S.b)
dL/dμ²ᴰ_curr.y += −(hx·S.b + hy·S.c)
```

`dL/dμ²ᴰ_curr` is **added to the same `dL_dmean2D` buffer** colour writes to.
Three genuinely new per-Gaussian accumulators are required:

- `dL_dprev_mean2D`   (float2)
- `dL_dsqrt_conic`    (float3, w.r.t. `S`)
- `dL_dprev_sqrt_cov2D` (float3, w.r.t. `P`)

Because the off-diagonal of each symmetric matrix is a *single* stored parameter
(`.b` appears in two matrix slots but is one float), both forward and backward treat
it as one variable — there is no factor-of-two bookkeeping at this stage. (The
factor-of-two for symmetric off-diagonals lives later, inside the cov2D→cov3D step,
backward.cu:225-227.)

A single fused **`diffFlowRenderBackwardCUDA`** kernel should emit colour gradients
(identical to `renderCUDA`) *and* the three new flow buffers, reading
`dL_dpixels` (colour) and `dL_dflows` (`[2,H,W]`).

### 2.2 `computeSquareRootCov2D` backward (the hard part, made easy)

Using the closed form of §1.4 (`X=(a+s)/t`, `Y=b/t`, `Z=(c+s)/t`), differentiate
analytically. With `det = ac − b²`, `s = √det`, `tr = a+c`, `t = √(tr+2s)`:

```
ds/da =  c/(2s)        ds/db = −b/s        ds/dc =  a/(2s)
dt/da = (1 + c/s)/(2t) dt/db = −b/(s·t)    dt/dc = (1 + a/s)/(2t)
```

For any output `O = N/t` (numerators `N_X=a+s`, `N_Y=b`, `N_Z=c+s`):

```
dO/dν = (dN/dν − O·dt/dν) / t          ν ∈ {a, b, c}
```

with
```
dN_X/da = 1 + ds/da   dN_X/db = ds/db     dN_X/dc = ds/dc
dN_Y/da = 0           dN_Y/db = 1         dN_Y/dc = 0
dN_Z/da = ds/da       dN_Z/db = ds/db     dN_Z/dc = 1 + ds/dc
```

Then accumulate the incoming `(X̄, Ȳ, Z̄) = dL/d√M`:

```
dL/da = X̄·dX/da + Ȳ·dY/da + Z̄·dZ/da
dL/db = X̄·dX/db + Ȳ·dY/db + Z̄·dZ/db
dL/dc = X̄·dX/dc + Ȳ·dY/dc + Z̄·dZ/dc
```

This gives a small `__device__ float3 computeSquareRootCov2DBackward(float3 cov2D,
float3 dL_dsqrt)` with no eigendecomposition and no `atan2` derivative.

> **Equivalent alternative:** solve the Lyapunov equation `S·M̄ + M̄·S = S̄`
> (the adjoint of `M = S²`). For symmetric 2×2 this is a 3×3 linear solve and gives
> the same result; the closed form above is cheaper and branch-free.

This backward is applied **twice**:

- Current frame: input is the **conic** (`conic_opacity.{x,y,z}`), output
  `dL/dconic` is *added to the colour `dL_dconic` buffer* before the existing
  `computeCov2DCUDA` runs — so the current-frame `S` path folds entirely into the
  existing cov2D→cov3D→{scale,rot,mean3D} machinery for free.
- Previous frame: input is the **plain prev cov2D** (the `prev_cov2D` buffer),
  output `dL/dprev_cov2D` feeds a *new* previous-frame cov2D backward (§2.3).

### 2.3 Previous-frame parameter path (entirely new machinery)

The current frame reuses `BACKWARD::preprocess` (backward.cu:559). The previous
frame has **no** existing backward and needs its own three steps:

1. **`dL/dprev_sqrt_cov2D → dL/dprev_cov2D`** via §2.2 with `prev_cov2D` as input.

2. **`dL/dprev_cov2D → dL/dprev_cov3D (+ dL/dprev_mean3D)`** — a variant of
   `computeCov2DCUDA` (backward.cu:144). Crucial difference: `computeCov2DCUDA`
   first converts `dL/dconic → dL/d{a,b,c}` through the matrix-inverse Jacobian
   (the `denom2inv` block, backward.cu:205-212). For the previous frame the stored
   quantity is the **plain** cov2D, so that inversion block is **skipped** — set
   `dL_da, dL_db, dL_dc` directly from `dL/dprev_cov2D.{x,y,z}` and continue from
   backward.cu:214 onward (the `T = W·J`, `cov2D = Tᵀ·Vrkᵀ·T` mapping) using
   `prev_means3D` and `prev` cov3D.

3. **`dL/dprev_cov3D → dL/dprev_scale, dL/dprev_rot`** via the existing
   `computeCov3D` device fn (backward.cu:278), and
   **`dL/dprev_mean2D → dL/dprev_mean3D`** via the projection block of
   `preprocessCUDA` (backward.cu:373-387). No SH term (the previous frame carries no
   colour). A dedicated `preprocessFlowBackwardCUDA` should bundle steps 2–3.

```
dL_dprev_sqrt_cov2D ─[§2.2]→ dL_dprev_cov2D ─[2]→ dL_dprev_cov3D ─[3]→ dL_dprev_scale
                                              └──→ dL_dprev_mean3D       dL_dprev_rot
dL_dprev_mean2D ───────────────[projection]──────→ dL_dprev_mean3D
```

---

## 3. Pre-existing forward bugs found (must fix before backward is meaningful)

While tracing the forward I found two bugs in `preprocessFlowCUDA` that make the
previous-frame quantities wrong, which would invalidate any backward built on top of
them:

1. **Wrong source point for the previous centre** (forward.cu:1199):
   ```c
   float3 p_prev_orig = { orig_points[3*idx], orig_points[3*idx+1], orig_points[3*idx+2] };
   ```
   uses `orig_points` (current) instead of `prev_orig_points`. As written,
   `prev_means2D == means2D` always, so the zero-flow test passes *trivially* but the
   uniform-translation test cannot — the previous centre never moves.

2. **Previous cov3D overwrites the current cov3D buffer** (forward.cu:1226-1228):
   ```c
   computeCov3D(prev_scales[idx], scale_modifier, prev_rotations[idx], cov3Ds + idx*6);
   prev_cov3D = cov3Ds + idx*6;
   ```
   writes into `cov3Ds + idx*6` — the **same** slot the current cov3D was just written
   to (forward.cu:1213). The current cov2D is computed *before* the overwrite so the
   forward render is unaffected, but the persisted `cov3Ds` buffer ends up holding the
   **previous** cov3D. The current-frame backward (`computeCov2DCUDA`) reads `cov3Ds`
   expecting the *current* cov3D — so it would silently use the wrong matrix.

   **Fix:** give the previous cov3D its own buffer. Add `float* prev_cov3D` to
   `FlowState` (rasterizer_impl.h:49) and write previous cov3D there, leaving
   `GeometryState.cov3D` for the current frame. The backward needs both buffers
   anyway (step §2.3.2 needs the previous cov3D).

---

## 4. Buffers & signatures to add (implementation checklist)

| Layer | Change |
|-------|--------|
| `forward.cu` | fix bugs §3; store prev cov3D in its own buffer |
| `rasterizer_impl.h` | `FlowState`: add `float* prev_cov3D` |
| `backward.cu` | `computeSquareRootCov2DBackward` device fn; `diffFlowRenderBackwardCUDA` kernel; `addSqrtConicGrad` kernel (conic path, §2.2 current); `preprocessFlowBackwardCUDA` (prev cov2D→cov3D→scale/rot + prev mean2D→mean3D); `BACKWARD::DIFF_FLOW::{render,preprocess}` wrappers |
| `backward.h` | declare the two wrappers |
| `rasterizer.h` | `DiffFlowRasterizer::backward(...)` declaration |
| `rasterizer_impl.cu` | `DiffFlowRasterizer::backward(...)` orchestration |
| `rasterize_points.{h,cu}` | `RasterizeGaussiansWithFlowBackwardCUDA` bridge — allocate `dL_d{prev_means3D,prev_scales,prev_rotations,prev_cov3D}` plus the current-frame outputs, return them |
| `ext.cpp` | bind `rasterize_gaussians_with_flow_backward` |
| `__init__.py` | `_RasterizeGaussiansWithFlow.backward` calls the new binding and returns real grads in the 13-entry tuple instead of `None` for the four prev params |

### New per-Gaussian gradient buffers in the render backward

- `dL_dmean2D` (float3)  — **shared** with colour, gets the §2.1(b) `μ²ᴰ_curr` term
- `dL_dconic` (float4)   — **shared** with colour, gets §2.2(current) after `addSqrtConicGrad`
- `dL_dsqrt_conic` (float3)        — new
- `dL_dprev_mean2D` (float2)       — new
- `dL_dprev_sqrt_cov2D` (float3)   — new

---

## 5. Verification plan

Once implemented, validate with the three tests from `flow-forward-pass.md` plus a
gradient check that now includes the flow output:

1. **Zero-flow** (`prev_* == *`): flow ≡ 0 and **all** `dL/dprev_*` ≡ 0. With bug §3.1
   fixed this becomes a non-trivial check.
2. **Uniform translation** (shift all `prev_means3D` by a constant): foreground flow
   ≈ Δ, and `dL/dprev_means3D` should match a finite-difference reference.
3. **`torch.autograd.gradcheck`** on `rasterize_gaussians_with_flow` in double
   precision with a flow-only loss, then a colour+flow loss, `eps=1e-3`. Compare the
   analytic `computeSquareRootCov2DBackward` against a finite-difference of
   `computeSquareRootCov2D` in isolation first — that device fn is the highest-risk
   piece.

> **Note:** none of this can be compiled or run in the current macOS/CPU
> environment — the extension requires a CUDA GPU (`python setup.py install`).
> The derivations above are written to be checked by `gradcheck` on a CUDA machine.
