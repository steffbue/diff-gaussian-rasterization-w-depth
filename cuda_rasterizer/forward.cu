/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Helper function to read cache
__device__ void readCache(
	const uint32_t P,
	const uint32_t g_index,
	const uint32_t pix_id,
	const uint2 bbx_min,
	const uint2 bbx_max,
	const uint32_t cache_offset,
	const float* __restrict__ t_value,
	const float* __restrict__ g_value,
	float* out_T,
	float* out_G)
{
	// If the Gaussian index is out of bounds, return 0.0f (no contribution)
	if (g_index >= P)
	{
		*out_T = 0.0f;
		*out_G = 0.0f;
		return;
	}

	// Compute bounding box size
	uint2 bbx_size = { bbx_max.x - bbx_min.x, bbx_max.y - bbx_min.y };

	// If the bounding box is invalid or pixel id is outside the bounding box, return 0.0f
	bool invalid_bbx = (bbx_size.x == 0 || bbx_size.y == 0);
	bool outside_bbx = (pix_id < bbx_min.x + bbx_min.y * bbx_size.x || pix_id > bbx_max.x + bbx_max.y * bbx_size.x);

	if (invalid_bbx || outside_bbx)
	{
		*out_T = 0.0f;
		*out_G = 0.0f;
		return;
	}

	// Compute offset in cache arrays
	uint32_t offset = cache_offset;

	*out_T = t_value[offset + (pix_id - (bbx_min.x + bbx_min.y * bbx_size.x))];
	*out_G = g_value[offset + (pix_id - (bbx_min.x + bbx_min.y * bbx_size.x))];
}

// Helper function to write cache
__device__ void writeCache(
	const uint32_t P,
	const uint32_t g_index,
	const uint32_t pix_id,
	const uint2 bbx_min,
	const uint2 bbx_max,
	const uint32_t cache_offset,
	float* __restrict__ t_value,
	float* __restrict__ g_value,
	const float T,
	const float G)
{
	// If the Gaussian index is out of bounds, return
	if (g_index >= P)
	{
		return;
	}

	// Compute bounding box size
	uint2 bbx_size = { bbx_max.x - bbx_min.x, bbx_max.y - bbx_min.y };

	// If the bounding box is invalid or pixel id is outside the bounding box, return
	bool invalid_bbx = (bbx_size.x == 0 || bbx_size.y == 0);
	bool outside_bbx = (pix_id < bbx_min.x + bbx_min.y * bbx_size.x || pix_id > bbx_max.x + bbx_max.y * bbx_size.x);

	if (invalid_bbx || outside_bbx)
	{
		return;
	}

	// Compute offset in cache arrays
	uint32_t offset = cache_offset;

	t_value[offset + (pix_id - (bbx_min.x + bbx_min.y * bbx_size.x))] = T;
	g_value[offset + (pix_id - (bbx_min.x + bbx_min.y * bbx_size.x))] = G;
}


// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
// 	float D = 0.0f;  // Mean Depth
    float D = 15.0f;  // Median Depth. TODO: This is a hack setting max_depth to 15

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            // Mean depth:
//             float dep = collected_depth[j];
//             D += dep * alpha * T;

            // Median depth:
            if (T > 0.5f && test_T < 0.5)
			{
			    float dep = collected_depth[j];
				D = dep;
			}


			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;
	}
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCacheCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H, int P,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	const float* __restrict__ depth,
	const uint2* __restrict__ bbx_min,
	const uint2* __restrict__ bbx_max,
	const uint32_t* __restrict__ cache_offset,
	float* __restrict__ t_value,
	float* __restrict__ g_value)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];
	
	__shared__ uint2 collected_bbx_min[BLOCK_SIZE];
	__shared__ uint2 collected_bbx_max[BLOCK_SIZE];
	__shared__ uint32_t collected_cache_offset[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
// 	float D = 0.0f;  // Mean Depth
    float D = 15.0f;  // Median Depth. TODO: This is a hack setting max_depth to 15

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];


			collected_bbx_min[block.thread_rank()] = bbx_min[coll_id];
			collected_bbx_max[block.thread_rank()] = bbx_max[coll_id];
			collected_cache_offset[block.thread_rank()] = cache_offset[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            // Mean depth:
//             float dep = collected_depth[j];
//             D += dep * alpha * T;

            // Median depth:
            if (T > 0.5f && test_T < 0.5)
			{
			    float dep = collected_depth[j];
				D = dep;
			}

			const float T_in = T;
			const float G_in = exp(power);

			// Write T and G values to cache.
			writeCache(
				P,
				collected_id[j],
				pix_id,
				collected_bbx_min[j],
				collected_bbx_max[j],
				collected_cache_offset[j],
				t_value,
				g_value,
				T_in,
				G_in
			);

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

		}
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth);
}

void FORWARD::renderCache(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H, int P,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	const float* depth,
	const uint2* bbx_min,
	const uint2* bbx_max,
	const uint32_t* cache_offset,
	float* t_value,
	float* g_value)
{
	renderCacheCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H, P,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		depth,
		bbx_min,
		bbx_max,
		cache_offset,
		t_value,
		g_value);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}

// Forward method for computing the square root of a 2D covariance matrix
__device__ float3 computeSquareRootCov2D(const float3& cov2D)
{
	// Compute trace, difference and off-diagonal elements
	float trace = cov2D.x + cov2D.z;
	float difference = cov2D.x - cov2D.z;
	float off_diag = 2.0f * cov2D.y;
	float discriminant = trace * trace - off_diag * off_diag;

	// Compute eigenvalues
	float eps = 1e-6f;
	float lambda1 = 0.5f * (trace + sqrt(fmaxf(eps, discriminant)));
	float lambda2 = 0.5f * (trace - sqrt(fmaxf(eps, discriminant)));

	// Ensure eigenvalues are non-negative
	lambda1 = fmaxf(eps, lambda1);
	lambda2 = fmaxf(eps, lambda2);

	// Compute square root of eigenvalues
	float sqrt_lambda1 = fmaxf(eps, sqrt(lambda1));
	float sqrt_lambda2 = fmaxf(eps, sqrt(lambda2));

	// Compute eigenvectors
	float cos_theta, sin_theta;
	float theta = 0.5f * atan2f(off_diag, difference);
	cos_theta = cosf(theta);
	sin_theta = sinf(theta);

	// Recombine matrices: Q * diag(sqrt(lambda1, sqrt(lambda2))) * Q^T
	float3 sqrt_cov2D;
	
	float cos_theta2 = cos_theta * cos_theta;
	float sin_theta2 = sin_theta * sin_theta;
	float cos_sin_theta = cos_theta * sin_theta;

	sqrt_cov2D.x = cos_theta2 * sqrt_lambda1 + sin_theta2 * sqrt_lambda2;
	sqrt_cov2D.y = cos_sin_theta * (sqrt_lambda1 - sqrt_lambda2);
	sqrt_cov2D.z = sin_theta2 * sqrt_lambda1 + cos_theta2 * sqrt_lambda2;

	return sqrt_cov2D;
}

// Forward method for computing the previous pixel position from the current pixel position
__device__ float2 computePrevPos(const float2& pos, const float2& prev_means2D, const float2& means2D, const float3& sqrt_cov2D, const float3& prev_sqrt_cov2D)
{

	// Compute diff between pixel positions and mean2D
	float2 diff = { pos.x - means2D.x, pos.y - means2D.y };

	// Compute pixel position in standard normal distribution
	float2 norm_pos = { sqrt_cov2D.x * diff.x + sqrt_cov2D.y * diff.y,
	                    sqrt_cov2D.y * diff.x + sqrt_cov2D.z * diff.y };

	// Compute pixel position in previous normal distribution	
	float2 prev_pos = { prev_sqrt_cov2D.x * norm_pos.x + prev_sqrt_cov2D.y * norm_pos.y,
	                    prev_sqrt_cov2D.y * norm_pos.x + prev_sqrt_cov2D.z * norm_pos.y };
	prev_pos.x += prev_means2D.x;
	prev_pos.y += prev_means2D.y;

	// Return previous position
	return prev_pos;
}

// Helper function to fetch transmittance values from cache
__device__ float fetchTransmittanceValue(
	const uint32_t P,
	const uint32_t g_offset,
	const uint32_t pix_id,
	const uint2* __restrict__ bbx_min,
	const uint2* __restrict__ bbx_max,
	const uint32_t* __restrict__ t_offset,
	const float* __restrict__ t_value)
{
	// If the Gaussian index is out of bounds, return 1.0f (no transmittance)
	if (g_offset >= P)
	{
		return 1.0f;
	}

	// Fetch bounding box min/max for the Gaussian
	uint2 min = bbx_min[g_offset];
	uint2 max = bbx_max[g_offset];

	// Compute offset in transmittance array
	uint32_t offset = t_offset[g_offset];

	uint2 bbx_size = { max.x - min.x, max.y - min.y };

	// If the bounding box is invalid or pixel id is outside the bounding box, return 1.0f
	bool invalid_bbx = (bbx_size.x == 0 || bbx_size.y == 0);
	bool outside_bbx = (pix_id < min.x + min.y * bbx_size.x || pix_id > max.x + max.y * bbx_size.x);

	if (invalid_bbx || outside_bbx)
	{
		return 1.0f;
	}
	
	return t_value[offset + (pix_id - (min.x + min.y * bbx_size.x))];
}

__global__ void computeCacheLayoutCUDA(
	int P,
	const uint2* bbx_min,
	const uint2* bbx_max,
	uint32_t* out_cache_counts_per_gaussian)
{
	// Identify current thread index
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize cache count for this Gaussian to 0
	uint32_t cache_count = 0;

	// Compute bounding box size
	uint2 bbx_size = { bbx_max[idx].x - bbx_min[idx].x, bbx_max[idx].y - bbx_min[idx].y };

	// If the bounding box is invalid, return
	if (bbx_size.x == 0 || bbx_size.y == 0)
	{
		out_cache_counts_per_gaussian[idx] = 0;
		return;
	}

	// Compute number of pixels in the bounding box
	cache_count = bbx_size.x * bbx_size.y;

	// Store the cache count for this Gaussian
	out_cache_counts_per_gaussian[idx] = cache_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
flowRenderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int P, int W, int H,
	const float2* __restrict__ points_xy_image,
	const float2* __restrict__ prev_points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	const float4* __restrict__ prev_cov2D_opacity,
	const float3* __restrict__ sqrt_conic,
	const float3* __restrict__ prev_sqrt_cov2D,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	const float* __restrict__ depth,
	float* __restrict__ out_depth,
	const uint2* __restrict__ bbx_min,
	const uint2* __restrict__ bbx_max,
	const uint32_t* __restrict__ cache_offset,
	const float* __restrict__ t_value,
	const float* __restrict__ g_value,
	float* __restrict__ out_flow)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_depth[BLOCK_SIZE];

	__shared__ float2 collected_prev_xy[BLOCK_SIZE];
	__shared__ float3 collected_sqrt_conic[BLOCK_SIZE];
	__shared__ float3 collected_prev_sqrt_cov2D[BLOCK_SIZE];

	__shared__ uint2 collected_bbx_min[BLOCK_SIZE];
	__shared__ uint2 collected_bbx_max[BLOCK_SIZE];
	__shared__ uint32_t collected_cache_offset[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
// 	float D = 0.0f;  // Mean Depth
    float D = 15.0f;  // Median Depth. TODO: This is a hack setting max_depth to 15
	float2 flow_vector = { 0.0f, 0.0f };
	float flow_weights = 0.0f;
	float2 prev_pixf = { 0.0f, 0.0f };

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			collected_depth[block.thread_rank()] = depth[coll_id];

			collected_prev_xy[block.thread_rank()] = prev_points_xy_image[coll_id];
			collected_sqrt_conic[block.thread_rank()] = sqrt_conic[coll_id];
			collected_prev_sqrt_cov2D[block.thread_rank()] = prev_sqrt_cov2D[coll_id];

			collected_bbx_min[block.thread_rank()] = bbx_min[coll_id];
			collected_bbx_max[block.thread_rank()] = bbx_max[coll_id];
			collected_cache_offset[block.thread_rank()] = cache_offset[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

            // Mean depth:
//             float dep = collected_depth[j];
//             D += dep * alpha * T;

            // Median depth:
            if (T > 0.5f && test_T < 0.5)
			{
			    float dep = collected_depth[j];
				D = dep;
			}


			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;

			// Compute the flow vector by transforming the previous pixel position
			// to the current pixel position using Gaussian Whitening and De-whitening.
			prev_pixf = computePrevPos(
				pixf,
				collected_prev_xy[j],
				collected_xy[j],
				collected_sqrt_conic[j],
				collected_prev_sqrt_cov2D[j]);

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001) for gaussian at t-1
			float2 prev_xy = collected_prev_xy[j];
			float2 prev_d = { prev_xy.x - prev_pixf.x, prev_xy.y - prev_pixf.y };
			float4 prev_con_o = prev_cov2D_opacity[j];
			float prev_power = -0.5f * (prev_con_o.x * prev_d.x * prev_d.x + prev_con_o.z * prev_d.y * prev_d.y) - prev_con_o.y * prev_d.x * prev_d.y;
			if (prev_power > 0.0f)
				continue;

			// Obtain alpha by multiplying with Gaussian opacity for gaussian at t-1
			float prev_alpha = min(0.99f, prev_con_o.w * exp(prev_power));
			if (prev_alpha < 1.0f / 255.0f)
				continue;
			float prev_test_T = T * (1 - prev_alpha);
			if (prev_test_T < 0.0001f)
			{
				continue;
			}

			float T = test_T;
			float G = exp(power);

			// Read T and G values from cache.
			float T_prev, G_prev;
			readCache(
				P,
				collected_id[j],
				pix_id,
				collected_bbx_min[j],
				collected_bbx_max[j],
				collected_cache_offset[j],
				t_value,
				g_value,
				&T_prev,
				&G_prev);


			// Compute weights (T, power and alpha) for t-1 and t
			float weights = T_prev * prev_alpha * G_prev * T * alpha * G;

			// Compute the flow vector by subtracting the previous pixel position
			// from the current pixel position.
			flow_vector.x += weights * (prev_pixf.x - pixf.x);
			flow_vector.y += weights * (prev_pixf.y - pixf.y);

			// Add weights to the flow weights
			flow_weights += weights;
				
		}
	}

	

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		// Normalize the flow vector by dividing by flow weights (soft assignment)
		if (flow_weights > 0.0f)
		{
			flow_vector.x /= flow_weights;
			flow_vector.y /= flow_weights;
		}
		else
		{
			// If no flow vector was computed, set it to zero
			flow_vector = { 0.0f, 0.0f };
		}

		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
		out_depth[pix_id] = D;

		// Write out the flow vector to the output buffer
		out_flow[pix_id] = flow_vector.x;
		out_flow[H * W + pix_id] = flow_vector.y;
	}
}

void FORWARD::FLOW::computeCacheLayout(
	int P,
	const uint2* bbx_min,
	const uint2* bbx_max,
	uint32_t* out_cache_counts_per_gaussian) 
{
	computeCacheLayoutCUDA<< <(P + 255) / 256, 256 >> > (
		P,
		bbx_min,
		bbx_max,
		out_cache_counts_per_gaussian);
}

void FORWARD::FLOW::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int P, int W, int H,
	const float2* means2D,
	const float2* prev_means2D,
	const float* colors,
	const float4* conic_opacity,
	const float4* prev_cov2D_opacity,
	const float3* sqrt_conic,
	const float3* prev_sqrt_cov2D,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	const float* depth,
	float* out_depth,
	const uint2* bbx_min,
	const uint2* bbx_max,
	const uint32_t* cache_offset,
	const float* t_value,
	const float* g_value,
	float* out_flow)
{
	flowRenderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		P, W, H,
		means2D,
		prev_means2D,
		colors,
		conic_opacity,
		prev_cov2D_opacity,
		sqrt_conic,
		prev_sqrt_cov2D,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		depth,
		out_depth,
		bbx_min,
		bbx_max,
		cache_offset,
		t_value,
		g_value,
		out_flow);
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessFlowCUDA(int P, int D, int M,
	const float* orig_points,
	const float* prev_orig_points,
	const glm::vec3* scales,
	const glm::vec3* prev_scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::vec4* prev_rotations,
	const float* opacities,
	const float* prev_opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* prev_cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float2* prev_points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	float4* prev_cov2D_opacity,
	float3* sqrt_conic,
	float3* prev_sqrt_cov2D,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// Transform previous point by projecting
	float3 p_prev_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_prev_hom = transformPoint4x4(p_prev_orig, projmatrix);
	float p_prev_w = 1.0f / (p_prev_hom.w + 0.0000001f);
	float3 p_prev_proj = { p_prev_hom.x * p_prev_w, p_prev_hom.y * p_prev_w, p_prev_hom.z * p_prev_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// If previous 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters.
	const float* prev_cov3D;
	if (prev_cov3D_precomp != nullptr)
	{
		prev_cov3D = prev_cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(prev_scales[idx], scale_modifier, prev_rotations[idx],
			cov3Ds + idx * 6);
		prev_cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Compute 2D screen-space covariance matrix for previous point
	float3 prev_cov = computeCov2D(p_prev_orig, focal_x, focal_y, tan_fovx, tan_fovy, prev_cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 
	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	getRect(point_image, my_radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
	radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	prev_points_xy_image[idx] = { ndc2Pix(p_prev_proj.x, W), ndc2Pix(p_prev_proj.y, H) };
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = { conic.x, conic.y, conic.z, opacities[idx] };
	prev_cov2D_opacity[idx] = { prev_cov.x, prev_cov.y, prev_cov.z, prev_opacities[idx] };
	sqrt_conic[idx] = computeSquareRootCov2D(conic);
	prev_sqrt_cov2D[idx] = computeSquareRootCov2D(prev_cov);
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	
}

void FORWARD::FLOW::preprocess(int P, int D, int M,
	const float* means3D,
	const float* prev_means3D,
	const glm::vec3* scales,
	const glm::vec3* prev_scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const glm::vec4* prev_rotations,
	const float* opacities,
	const float* prev_opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* prev_cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float2* prev_means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	float4* prev_cov2D,
	float3* sqrt_conic,
	float3* prev_sqrt_cov2D,
	bool prefiltered)
{
	preprocessFlowCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		prev_means3D,
		scales,
		prev_scales,
		scale_modifier,
		rotations,
		prev_rotations,
		opacities,
		prev_opacities,
		shs,
		clamped,
		cov3D_precomp,
		prev_cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		prev_means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prev_cov2D,
		sqrt_conic,
		prev_sqrt_cov2D,
		prefiltered
		);
}