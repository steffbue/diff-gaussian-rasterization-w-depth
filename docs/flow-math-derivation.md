# Optical Flow — Mathematical Derivation (Forward & Backward)

This document is the **self-contained mathematical derivation** of the differentiable
optical-flow renderer (`GaussianRasterizerWithFlow` / `DiffFlowRasterizer`). It states
every map in the pipeline and derives its adjoint (vector–Jacobian product) from first
principles. Every result here is verified against finite differences to ≈1e-11.

For *design rationale* and *where the code lives*, read
[`flow-rasterizer-walkthrough.md`](flow-rasterizer-walkthrough.md),
[`flow-forward-pass.md`](flow-forward-pass.md) and
[`flow-backward-design.md`](flow-backward-design.md). This file is the maths only.

---

## 0. Notation and conventions

| Symbol | Meaning |
|--------|---------|
| `p ∈ ℝ²` | pixel centre (screen coordinates, pixel units), a **constant** during diff. |
| `i` | Gaussian index, streamed front-to-back per tile. |
| `μᵢ ∈ ℝ²` | projected 2D centre of Gaussian `i`, current frame (`means2D`). |
| `μ'ᵢ ∈ ℝ²` | projected 2D centre, previous frame (`prev_means2D`). |
| `Σᵢ ∈ 𝕊²₊` | 2D covariance (current), `Σ'ᵢ` previous. SPD symmetric 2×2. |
| `Σᵢ⁻¹` | the *conic* (current frame; `conic_opacity.xyz`). |
| `Sᵢ = √(Σᵢ⁻¹)` | matrix sqrt of the conic (`sqrt_conic`). |
| `Pᵢ = √(Σ'ᵢ)` | matrix sqrt of the plain previous cov2D (`prev_sqrt_cov2D`). |
| `oᵢ ∈ (0,1]` | opacity (current frame only). |
| `αᵢ, Tᵢ` | per-pixel alpha and transmittance, `wᵢ = αᵢ Tᵢ` the compositing weight. |
| `cᵢ ∈ ℝ³` | view-dependent colour (current frame). |
| `Δᵢ(p) ∈ ℝ²` | per-pixel displacement induced by Gaussian `i` (the flow "feature"). |

A symmetric 2×2 matrix is stored as a `float3 (a, b, c) = [[a, b],[b, c]]`. **The
off-diagonal `b` is a single scalar parameter** (it appears in two matrix slots but is
one stored value); this matters for factor-of-two bookkeeping in the adjoints (§2.4).

The loss `L` is scalar. For a quantity `x` we write its adjoint `x̄ = ∂L/∂x`. Incoming
pixel gradients are `ḡ_C = ∂L/∂C(p) ∈ ℝ³` (colour) and `ḡ_F = ∂L/∂flow(p) ∈ ℝ²` (flow).

---

## 1. Forward pass

### 1.1 Geometry: 3D → 2D covariance

Each Gaussian's 3D covariance comes from scale `s` and rotation quaternion `q`:

```
M = S R,    Σ₃D = Mᵀ M        (computeCov3D)
```

with `S = diag(s)` and `R` the rotation matrix of `q`. Projection to the image plane uses
the EWA splatting affine approximation (Zwicker et al. 2002). With view matrix `W`
(3×3 rotation block), the Jacobian `J` of the perspective map at the (frustum-clamped)
view-space point `t`, and `T = W J`:

```
Σ₂D = Tᵀ Σ₃Dᵀ T,   then  Σ₂D[0,0] += 0.3,  Σ₂D[1,1] += 0.3      (computeCov2D)
```

The `+0.3` diagonal low-pass guarantees each Gaussian covers ≥1 pixel. The current frame
then **inverts** `Σ₂D` to the conic `Σ₂D⁻¹` and packs `conic_opacity = (Σ₂D⁻¹, o)`. The
previous frame keeps the **plain** `Σ'₂D` (no inversion). This asymmetry — current stores
the inverse, previous stores the plain — is exactly what whitening/de-whitening needs
(§1.4).

### 1.2 The matrix square root and its key identity

For a symmetric **PSD** 2×2 matrix `M = [[a,b],[b,c]]`, the kernel computes `√M` by
eigendecomposition (`computeSquareRootCov2D`), but this is **analytically identical** to
the branch-free closed form

```
s = √det(M),   det = a c − b²
t = √(tr(M) + 2s),   tr = a + c
√M = (M + s I) / t   ⟹   X = (a+s)/t,  Y = b/t,  Z = (c+s)/t.        (★)
```

**Proof.** Let `M` have eigenvalues `λ₁, λ₂ ≥ 0` with orthonormal eigenvectors. Then
`s = √(λ₁λ₂) = √λ₁·√λ₂` and `tr + 2s = λ₁ + λ₂ + 2√λ₁√λ₂ = (√λ₁ + √λ₂)²`, so
`t = √λ₁ + √λ₂ = tr(√M)`. In the eigenbasis of `M`, `(M + sI)` has eigenvalue
`λₖ + √λ₁√λ₂ = √λₖ(√λ₁ + √λ₂) = √λₖ · t`, hence `(M+sI)/t` has eigenvalue `√λₖ` on the
same eigenvector — i.e. it equals `√M`. ∎

Identity (★) is what makes the backward tractable; its adjoint is derived in §2.4. (The
forward's `eps`-clamps inside the eigendecomposition are dropped in the backward, standard
practice.)

### 1.3 Alpha compositing (colour, depth, flow share one weight)

Streaming front-to-back with `T₁ = 1`, for each contributing Gaussian:

```
αᵢ = min(0.99, oᵢ · exp(powerᵢ)),   powerᵢ = −½ (p−μᵢ)ᵀ Σᵢ⁻¹ (p−μᵢ)
Tᵢ₊₁ = Tᵢ (1 − αᵢ)
wᵢ  = αᵢ Tᵢ
```

The three outputs use **the same** `wᵢ`:

```
C(p)    = Σᵢ wᵢ cᵢ  +  T_{N+1} · bg            (colour, with background)
flow(p) = Σᵢ wᵢ Δᵢ(p)                          (no background, no normalisation)
depth(p)= z_k  where k is the first i with  Tᵢ > 0.5 ≥ Tᵢ₊₁   (median depth; default 15)
```

Using identical weights for colour and flow is the entire design idea: the flow output is
just "two more channels," so its backward is one extra term in the colour machinery
(§2.2). Background pixels (`Σ wᵢ → 0`) receive zero flow, consistent with colour. **Depth
is a hard selection (argmax-style) and is treated as non-differentiable** — `grad_depth`
is ignored.

### 1.4 The displacement model `Δᵢ(p)` (`computePrevPos`)

`Δᵢ` is the per-pixel flow vector induced by Gaussian `i` having moved/scaled/rotated
between frames. With `Sᵢ = √(Σᵢ⁻¹)` and `Pᵢ = √(Σ'ᵢ)`:

```
diff      = p − μᵢ
norm_pos  = Sᵢ · diff                  # whiten: into the unit-variance frame of i (current)
prev_pos  = Pᵢ · norm_pos + μ'ᵢ        # de-whiten: out into i's previous frame
Δᵢ(p)     = prev_pos − p
```

**Why this is the right map.** Whitening sends `p` to `n = Σᵢ^{-1/2}(p−μᵢ)`, which has
identity covariance and Mahalanobis radius `‖n‖² = (p−μᵢ)ᵀ Σᵢ⁻¹ (p−μᵢ)`. De-whitening
sends `n` to `q = Σ'ᵢ^{1/2} n + μ'ᵢ`, whose Mahalanobis radius in the **previous**
Gaussian, `(q−μ'ᵢ)ᵀ Σ'ᵢ⁻¹ (q−μ'ᵢ) = nᵀn`, is **identical**. So the displacement maps
each iso-probability contour of the current Gaussian onto the matching contour of the
previous Gaussian — a translation, rotation and anisotropic scale of the splat, not just a
rigid centre shift. (Note `√(Σ⁻¹) = (√Σ)⁻¹` for SPD `Σ`, so `Sᵢ` whitens and `Pᵢ`
de-whitens correctly.) The composite linear part is `Aᵢ = Pᵢ Sᵢ = Σ'ᵢ^{1/2} Σᵢ^{-1/2}`.

Written out componentwise with `S=(Sₐ,S_b,S_c)`, `P=(Pₐ,P_b,P_c)`:

```
norm_pos.x = Sₐ·diff.x + S_b·diff.y
norm_pos.y = S_b·diff.x + S_c·diff.y
prev_pos.x = Pₐ·norm_pos.x + P_b·norm_pos.y + μ'.x
prev_pos.y = P_b·norm_pos.x + P_c·norm_pos.y + μ'.y
```

---

## 2. Backward pass

The backward replays the tile loop **back-to-front**, reconstructing `Tᵢ = Tᵢ₊₁/(1−αᵢ)`,
and accumulates gradients with `atomicAdd`. There are four derivations: the
alpha-compositing adjoint (§2.1), the two flow routes (§2.2–2.3), the displacement adjoint
(§2.3), and the matrix-sqrt adjoint (§2.4). The current frame then reuses the standard
EWA backward; the previous frame needs its own cov2D→cov3D→{scale,rot,mean} chain (§2.5).

### 2.1 Alpha-compositing adjoint (the recurrence)

Treat any output channel as `O(p) = Σᵢ wᵢ fᵢ + T_{N+1} f_bg` with feature `fᵢ` (colour
`cᵢ`, or flow `Δᵢ`; flow has `f_bg = 0`). Two facts:

- **Direct dependence on `αᵢ`:** `wᵢ = αᵢ Tᵢ`, and `Tᵢ` does *not* depend on `αᵢ`.
- **Indirect dependence:** every later weight contains the factor `(1−αᵢ)`, since
  `Tⱼ = ∏_{k<j}(1−αₖ)`. So `∂(αⱼTⱼ)/∂αᵢ = −αⱼTⱼ/(1−αᵢ)` for `j > i`, and
  `∂T_{N+1}/∂αᵢ = −T_{N+1}/(1−αᵢ)`.

Therefore

```
∂O/∂αᵢ = Tᵢ fᵢ  −  1/(1−αᵢ) · ( Σ_{j>i} αⱼTⱼ fⱼ  +  T_{N+1} f_bg ).        (†)
```

Define the **back-to-front suffix accumulator** `Aᵢ = (1/Tᵢ₊₁) Σ_{j>i} αⱼ Tⱼ fⱼ`, which
satisfies the cheap recurrence used in the kernel (`accum_rec`):

```
Aᵢ = α_{i+1} f_{i+1} + (1 − α_{i+1}) A_{i+1},   A_N = 0.
```

Since `Tᵢ₊₁ = Tᵢ(1−αᵢ)`, the middle term of (†) is `Tᵢ Aᵢ`. Hence the contribution to the
alpha gradient from this channel is

```
dL/dαᵢ  +=  Tᵢ · (fᵢ − Aᵢ) · f̄        (channel feature term)
```

and the background term (colour only) is `dL/dαᵢ += −T_{N+1}/(1−αᵢ) · (bg · ḡ_C)`. The
per-Gaussian feature itself gets `f̄ᵢ = wᵢ · f̄` (the direct `wᵢ fᵢ` term); for colour this
is `c̄ᵢ = αᵢTᵢ ḡ_C`, written straight to `dL_dcolors`.

The kernel maintains `accum_rec[ch]` for the 3 colour channels and `accum_rec_flow[0..1]`
for the 2 flow channels with the *same* recurrence, and folds **both** feature terms into
one `dL_dalpha`, multiplied by `Tᵢ` once. This is the sense in which "flow is two extra
colour channels."

### 2.2 Flow route (a): through the weight `αᵢ`

This is just §2.1 applied with `fᵢ = Δᵢ` and `f̄ = ḡ_F`:

```
dL/dαᵢ += Tᵢ · (Δᵢ.x − accum_rec_flowx)·ḡ_F.x  +  Tᵢ · (Δᵢ.y − accum_rec_flowy)·ḡ_F.y
```

added into the **same** `dL_dαᵢ` as colour. That combined `dL_dαᵢ` then flows through the
standard `α → G → {conic, mean2D, opacity}` block (§2.5), unchanged. No new machinery.

### 2.3 Flow route (b): through the displacement `Δᵢ` directly

The direct `wᵢ Δᵢ` term gives `Δ̄ᵢ = wᵢ ḡ_F`. The kernel calls this `q = αᵢTᵢ · ḡ_F`.
Since `Δᵢ = prev_pos − p` and `p` is constant, `∂L/∂prev_pos = q`. Back-propagating
through `computePrevPos` (§1.4) by plain chain rule:

```
# previous 2D centre   (prev_pos = … + μ')
μ̄'ᵢ      += q

# Pᵢ = √Σ'  (prev_pos = P·norm_pos + μ')
P̄ᵢ.a += q.x·norm_pos.x
P̄ᵢ.b += q.x·norm_pos.y + q.y·norm_pos.x
P̄ᵢ.c += q.y·norm_pos.y

# back through Pᵢ into norm_pos
h = ( q.x·Pₐ + q.y·P_b ,  q.x·P_b + q.y·P_c )

# Sᵢ = √Σ⁻¹  (norm_pos = S·diff)
S̄ᵢ.a += h.x·diff.x
S̄ᵢ.b += h.x·diff.y + h.y·diff.x
S̄ᵢ.c += h.y·diff.y

# back through Sᵢ into diff = p − μᵢ   (⟹ μ̄ᵢ = −∂L/∂diff)
μ̄ᵢ.x += −(h.x·Sₐ + h.y·S_b)
μ̄ᵢ.y += −(h.x·S_b + h.y·S_c)
```

Notes:
- `μ̄ᵢ` (current centre) is added into the **same `dL_dmean2D`** colour writes to.
- The off-diagonal accumulates **both** matrix slots into the single stored `.b`
  (`q.x·norm_pos.y + q.y·norm_pos.x`), because `b` is one parameter (§0). There is no
  factor-of-two *here*; the symmetric factor-of-two appears later, in cov2D→cov3D (§2.5).
- Three genuinely new per-Gaussian accumulators are needed: `dL_dsqrt_conic` (= `S̄`),
  `dL_dprev_mean2D` (= `μ̄'`), `dL_dprev_sqrt_cov2D` (= `P̄`).

*(Verified vs. finite difference: all four adjoints to ≤7e-11.)*

### 2.4 Matrix-square-root adjoint (`computeSquareRootCov2DBackward`)

Given the incoming `√M̄ = (X̄, Ȳ, Z̄)` and the closed form (★), differentiate analytically.
With `s = √det`, `tr = a+c`, `t = √(tr+2s)`:

```
∂s/∂a =  c/(2s)         ∂s/∂b = −b/s        ∂s/∂c =  a/(2s)
∂t/∂a = (1 + c/s)/(2t)  ∂t/∂b = −b/(s·t)    ∂t/∂c = (1 + a/s)/(2t)
```

Each output is `O = N/t` (`N_X=a+s`, `N_Y=b`, `N_Z=c+s`), so

```
∂O/∂ν = ( ∂N/∂ν − O · ∂t/∂ν ) / t ,   ν ∈ {a, b, c},
```

with numerator partials

```
∂N_X/∂(a,b,c) = (1+∂s/∂a, ∂s/∂b, ∂s/∂c)
∂N_Y/∂(a,b,c) = (0, 1, 0)
∂N_Z/∂(a,b,c) = (∂s/∂a, ∂s/∂b, 1+∂s/∂c)
```

and finally `M̄.ν = X̄·∂X/∂ν + Ȳ·∂Y/∂ν + Z̄·∂Z/∂ν`. No eigendecomposition, no `atan2`
derivative.

> **Equivalent view:** `M̄` is the solution of the Sylvester/Lyapunov equation
> `√M · M̄ + M̄ · √M = √M̄` (the adjoint of `M = (√M)²`). For symmetric 2×2 that is a 3×3
> linear solve giving the same answer; the closed form above is cheaper and branch-free.

*(Verified vs. finite difference to ≈7e-11.)*

This routine is applied **twice**:

1. **Current frame** — input is the conic `Σ⁻¹` (from `conic_opacity.xyz`), incoming
   `S̄ = dL_dsqrt_conic`. The result `dL/d(Σ⁻¹)` is **added into the colour `dL_dconic`
   buffer** (`addSqrtConicGradCUDA`) *before* the standard `computeCov2DCUDA` runs, so the
   current-frame `S` path folds for free into the existing conic→cov2D→cov3D→{scale,rot,
   mean3D} machinery. (The colour path already produced `dL_dconic` from the
   alpha/`power` term; flow just adds to it.)
2. **Previous frame** — input is the plain `Σ'₂D` (`prev_cov2D`), incoming
   `P̄ = dL_dprev_sqrt_cov2D`. The result `dL/dΣ'₂D` feeds the new previous-frame chain
   (§2.5).

### 2.5 Mapping 2D gradients to 3D parameters

**Current frame (reused, unchanged).** The combined `dL_dmean2D`, `dL_dconic` (now
including the flow contributions of §2.3 and §2.4-current) and `dL_dopacity` go through
the *standard* colour backward:

- `α → G → {conic, mean2D, opacity}`: with `G = exp(power)`, `dL/dG = oᵢ·dL/dαᵢ`,
  `dL/do = G·dL/dαᵢ`, and the conic / `power` partials
  `∂G/∂conic = −½ G (p−μ)(p−μ)ᵀ`, `∂G/∂μ` (the `dG_ddel·ddel_dx` block, with
  `ddelx_dx = ½W`, `ddely_dy = ½H` from the NDC→pixel map).
- `conic = Σ₂D⁻¹`: the matrix-inverse Jacobian (`denom2inv` block) maps `dL/dconic →
  dL/dΣ₂D`.
- `Σ₂D = Tᵀ Σ₃Dᵀ T` and `T = W J`: maps `dL/dΣ₂D → dL/dΣ₃D` and `dL/dμ_view` (mean3D),
  with the symmetric off-diagonal factor-of-two appearing here.
- `Σ₃D = (SR)ᵀ(SR)` (`computeCov3D` backward): `dL/dΣ₃D → dL/dscale, dL/drotation`.
- mean2D projection backward: `dL/dmean2D → dL/dmean3D`.

**Previous frame (new chain).** The previous frame has no opacity, colour, SH or conic, so
its chain starts from `dL/dΣ'₂D` and `dL/dμ'`:

```
dL_dprev_sqrt_cov2D ─[§2.4]→ dL_dprev_cov2D ─[A]→ dL_dprev_cov3D ─[B]→ dL_dprev_scale
                                              └──→ dL_dprev_mean3D       dL_dprev_rot
dL_dprev_mean2D ──────────────[projection, C]─────→ dL_dprev_mean3D  (+=)
```

- **[A] `dL/dΣ'₂D → dL/dΣ'₃D (+ dL/dμ'₃D)`** (`computeCov2DFlowBackwardCUDA`): a clone of
  `computeCov2DCUDA` that **omits the conic-inverse Jacobian** (the previous frame stores
  the plain `Σ'₂D`, not its inverse), then applies the same `Σ₂D = Tᵀ Σ₃Dᵀ T`,
  `T = W J` mapping using `prev_means3D` and the previous cov3D (`FlowState::prev_cov3D`).
  This produces `dL_dprev_cov3D` and **assigns** the EWA-Jacobian part of
  `dL_dprev_means3D`.
- **[B] `dL/dΣ'₃D → dL/dprev_scale, dL/dprev_rot`** via `computeCov3D` backward; and
- **[C] `dL/dμ' → dL/dprev_means3D`** via the projection backward — added (`+=`) onto the
  mean3D gradient that [A] assigned (`preprocessFlowBackwardCUDA`). No SH term (the
  previous frame carries no colour).

> **Ordering invariants.** `addSqrtConicGradCUDA` must run **before** `computeCov2DCUDA`
> (so the folded `dL_dconic` is seen). `computeCov2DFlowBackwardCUDA` (which *assigns*
> `dL_dprev_means3D`) must run **before** `preprocessFlowBackwardCUDA` (which *adds* the
> projection part). Reordering silently drops gradient.

---

## 3. Adjoint summary (all routes into each parameter)

| Parameter | Receives gradient via |
|-----------|----------------------|
| `colors`/`sh` | direct colour term `wᵢ ḡ_C` (+ SH→colour backward) |
| `opacities` | `dL/do = G·dL/dαᵢ` (αᵢ carries colour **and** flow route (a)) |
| `means2D`/`means3D` (curr) | colour `α→μ` block **+** flow route (b) `μ̄ᵢ` term |
| `scales`/`rotations` (curr) | conic path: colour conic + flow `S̄` (§2.4-current) → cov3D |
| `cov3Ds_precomp` (curr) | same conic path, stopping at cov3D |
| `prev_means3D` | flow route (b) `μ̄'` (proj.) **+** EWA-Jacobian of `dL/dΣ'₂D` |
| `prev_scales`/`prev_rotations` | flow `P̄` (§2.4-prev) → `Σ'₂D` → `Σ'₃D` → scale/rot |
| `prev_cov3Ds_precomp` | same, stopping at `Σ'₃D` |
| depth | **none** (median selection is non-differentiable; `grad_depth` ignored) |

### Sanity checks the maths must satisfy (encoded in `tests/test_flow_backward.py`)

1. **Zero-flow:** `prev_* == curr_*` ⟹ `Δᵢ ≡ 0` ⟹ `flow ≡ 0` and **all** `dL/dprev_* ≡ 0`.
   (Non-trivial only after the forward bug — prev centre projected from `orig_points` —
   was fixed; see walkthrough Gotcha #2.)
2. **Uniform translation:** shift all `prev_means3D` by a constant ⟹ foreground flow ≈ Δ;
   `dL/dprev_means3D` matches finite difference.
3. **Finite difference** on `means3D` and `prev_means3D` with a flow-only and a colour+flow
   loss. Kernels are float32, so use loose-tolerance finite differences, **not**
   `torch.autograd.gradcheck` (float64).

---

## 4. One-line recap

Colour, depth and flow share the single compositing weight `wᵢ = αᵢTᵢ`. Flow's feature is
the whitening/de-whitening displacement `Δᵢ(p) = Σ'ᵢ^{1/2} Σᵢ^{-1/2}(p−μᵢ) + μ'ᵢ − p`.
Backward = colour backward **plus** (a) `Δᵢ` as two extra compositing channels into
`dL/dαᵢ`, and (b) `Δ̄ᵢ = wᵢ ḡ_F` routed through `computePrevPos`'s linear maps and the two
matrix-sqrt adjoints — the current-frame branch folding into the existing EWA chain, the
previous-frame branch using its own conic-free cov2D→cov3D→{scale,rot,mean} chain.