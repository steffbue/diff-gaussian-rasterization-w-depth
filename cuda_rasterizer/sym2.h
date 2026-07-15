/*
 * Symmetric-2x2 / 2-vector algebra for the optical-flow math.
 *
 * The flow forward/backward passes (see docs/flow-math-derivation.md) are built
 * almost entirely out of a handful of operations on 2-vectors and symmetric 2x2
 * matrices: whiten a pixel into a Gaussian's unit frame, de-whiten it into
 * another, take a matrix square root, and the adjoints of all of these.
 *
 * Previously each of these was written out as raw scalar arithmetic on float3
 * components, separately in forward.cu and backward.cu. This header collects the
 * primitives so that (a) each formula lives in exactly one place and (b) the
 * backward reads as the literal transpose/adjoint of the forward.
 *
 * Conventions
 * -----------
 * A symmetric 2x2 matrix  M = [[a, b], [b, c]]  is stored as float3 {a, b, c}
 * (the exact packing already used for sqrt_conic / cov2D). The off-diagonal `b`
 * is ONE scalar parameter (it occupies two matrix slots but is a single stored
 * value); the adjoint helpers below fold both slots into that one value, so call
 * sites need no factor-of-two bookkeeping.
 *
 * Performance
 * -----------
 * Every helper is __forceinline__ __device__ on POD types, so the compiler
 * inlines each to exactly the scalar code it replaces — zero runtime cost.
 */

#ifndef CUDA_RASTERIZER_SYM2_H_INCLUDED
#define CUDA_RASTERIZER_SYM2_H_INCLUDED

#include <cuda_runtime.h>

// Semantic aliases (no layout change: these ARE float2/float3).
using Vec2 = float2;   // 2-vector (x, y)
using Sym2 = float3;   // symmetric 2x2 matrix {a=xx, b=xy=yx, c=yy}

// --- 2-vector helpers ------------------------------------------------------
__forceinline__ __device__ Vec2 v_add(const Vec2& a, const Vec2& b) { return { a.x + b.x, a.y + b.y }; }
__forceinline__ __device__ Vec2 v_sub(const Vec2& a, const Vec2& b) { return { a.x - b.x, a.y - b.y }; }
__forceinline__ __device__ Vec2 v_neg(const Vec2& a)                { return { -a.x, -a.y }; }

// --- symmetric matrix · vector:  y = M v -----------------------------------
__forceinline__ __device__ Vec2 symv(const Sym2& M, const Vec2& v)
{
	return { M.x * v.x + M.y * v.y,
	         M.y * v.x + M.z * v.y };
}

// --- adjoint of (y = M v) w.r.t. the symmetric matrix M --------------------
// For incoming dL/dy = q, the gradient w.r.t. M is the symmetric outer product
// of q and v, with the single off-diagonal accumulating both slots:
//   dL/da = q.x·v.x,  dL/db = q.x·v.y + q.y·v.x,  dL/dc = q.y·v.y.
// (The adjoint w.r.t. v is just M·q, since M is symmetric — use symv(M, q).)
__forceinline__ __device__ Sym2 sym_outer(const Vec2& q, const Vec2& v)
{
	return { q.x * v.x,
	         q.x * v.y + q.y * v.x,
	         q.y * v.y };
}

// eps floor for `det` inside sqrt_sym2/sqrt_sym2_vjp exists only to guard
// against floating-point noise pushing a near-singular (truly ~0) det
// slightly negative before the sqrtf -- it must NOT engage for a legitimate,
// well-conditioned PSD matrix that simply has a small *absolute* det. That
// happens routinely for `conic` (= inverse of the projected 2D covariance):
// det(conic) = 1/det(cov), which is already <1e-6 for any moderately large
// (not even degenerate) on-screen covariance, e.g. det(cov) > 1e6. A fixed
// absolute eps=1e-6f (as before) silently clamps `s`/`t` upward in exactly
// that routine case, corrupting sqrt_conic (and its gradient) even though
// the input matrix was perfectly well-conditioned -- confirmed on a real
// trained scene: sqrt_sym2(conic) stopped satisfying sqrt(conic)^2==conic
// once det(cov) exceeded ~1e6, while remaining exact in float64 once this
// eps is scaled relative to the matrix's own magnitude instead of fixed.
// Two floors are needed, in different units: `det` has units of value^2
// (a*c), `tr + 2s` has units of value^1 (a+c). Both scaled relative to `tr`
// so neither engages for a legitimate, merely-small-magnitude matrix --
// only for genuine near-zero/negative floating-point noise around a truly
// singular input.
__forceinline__ __device__ float sym2DetEps(float tr) { return fmaxf(1e-30f, tr * tr * 1e-12f); }
__forceinline__ __device__ float sym2TEps(float tr)   { return fmaxf(1e-15f, tr * 1e-6f); }

// --- matrix square root of a symmetric PSD 2x2 (closed form) ---------------
// For PSD symmetric M this equals the eigendecomposition-based sqrt but is
// branch-free and cheaper:  s = sqrt(det),  t = sqrt(tr + 2s),  sqrt(M) = (M + sI)/t.
__forceinline__ __device__ Sym2 sqrt_sym2(const Sym2& M)
{
	const float tr = M.x + M.z;
	const float det = fmaxf(sym2DetEps(tr), M.x * M.z - M.y * M.y);
	const float s = sqrtf(det);
	const float t = sqrtf(fmaxf(sym2TEps(tr), tr + 2.0f * s));
	return { (M.x + s) / t, M.y / t, (M.z + s) / t };
}

// --- adjoint (VJP) of sqrt_sym2 --------------------------------------------
// Given dL/d(sqrt(M)) = dL_dsqrt, return dL/dM. Differentiates the closed form
// above analytically (no eigendecomposition). With outputs X=(a+s)/t, Y=b/t,
// Z=(c+s)/t and O=N/t we use dO/dv = (dN/dv - O·dt/dv)/t for v in {a,b,c}.
__forceinline__ __device__ Sym2 sqrt_sym2_vjp(const Sym2& M, const Sym2& dL_dsqrt)
{
	const float a = M.x;
	const float b = M.y;
	const float c = M.z;

	const float tr = a + c;
	const float det = fmaxf(sym2DetEps(tr), a * c - b * b);
	const float s = sqrtf(det);
	const float t = sqrtf(fmaxf(sym2TEps(tr), tr + 2.0f * s));

	// Forward outputs
	const float X = (a + s) / t;
	const float Y = b / t;
	const float Z = (c + s) / t;

	// Partials of s and t w.r.t. (a, b, c)
	const float ds_da = c / (2.0f * s);
	const float ds_db = -b / s;
	const float ds_dc = a / (2.0f * s);

	const float dt_da = (1.0f + c / s) / (2.0f * t);
	const float dt_db = -b / (s * t);
	const float dt_dc = (1.0f + a / s) / (2.0f * t);

	// Numerator partials: N_X = a + s, N_Y = b, N_Z = c + s
	const float dNx_da = 1.0f + ds_da, dNx_db = ds_db, dNx_dc = ds_dc;
	const float dNy_da = 0.0f,         dNy_db = 1.0f,  dNy_dc = 0.0f;
	const float dNz_da = ds_da,        dNz_db = ds_db, dNz_dc = 1.0f + ds_dc;

	// For O = N / t :  dO/dv = (dN/dv - O * dt/dv) / t
	const float dX_da = (dNx_da - X * dt_da) / t;
	const float dX_db = (dNx_db - X * dt_db) / t;
	const float dX_dc = (dNx_dc - X * dt_dc) / t;

	const float dY_da = (dNy_da - Y * dt_da) / t;
	const float dY_db = (dNy_db - Y * dt_db) / t;
	const float dY_dc = (dNy_dc - Y * dt_dc) / t;

	const float dZ_da = (dNz_da - Z * dt_da) / t;
	const float dZ_db = (dNz_db - Z * dt_db) / t;
	const float dZ_dc = (dNz_dc - Z * dt_dc) / t;

	Sym2 dL_dM;
	dL_dM.x = dL_dsqrt.x * dX_da + dL_dsqrt.y * dY_da + dL_dsqrt.z * dZ_da;
	dL_dM.y = dL_dsqrt.x * dX_db + dL_dsqrt.y * dY_db + dL_dsqrt.z * dZ_db;
	dL_dM.z = dL_dsqrt.x * dX_dc + dL_dsqrt.y * dY_dc + dL_dsqrt.z * dZ_dc;
	return dL_dM;
}

#endif