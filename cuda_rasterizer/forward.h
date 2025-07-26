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

#ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
#define CUDA_RASTERIZER_FORWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace FORWARD
{
	// Perform initial steps for each Gaussian prior to rasterization.
	void preprocess(int P, int D, int M,
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
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		int* radii,
		float2* points_xy_image,
		float* depths,
		float* cov3Ds,
		float* colors,
		float4* conic_opacity,
		const dim3 grid,
		uint32_t* tiles_touched,
		bool prefiltered);

	// Main rasterization method.
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float2* points_xy_image,
		const float* features,
		const float4* conic_opacity,
		float* final_T,
		uint32_t* n_contrib,
		const float* bg_color,
		float* out_color,
		const float* depth,
	    float* out_depth);

	namespace FLOW 
	{
		// Perform initial steps for each Gaussian prior to rasterization.
		void preprocess(int P, int D, int M,
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
			const float focal_x, float focal_y,
			const float tan_fovx, float tan_fovy,
			int* radii,
			float2* points_xy_image,
			float2* prev_points_xy_image,
			float* depths,
			float* cov3Ds,
			float* colors,
			float4* conic_opacity,
			const dim3 grid,
			uint32_t* tiles_touched,
			float4* prev_cov2D_opacity,
			float3* sqrt_conic,
			float3* prev_sqrt_cov2D,
			bool prefiltered);

		// Main rasterization method.
		void render(
			const dim3 grid, dim3 block,
			const uint2* ranges,
			const uint32_t* point_list,
			int W, int H,
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
			float* out_flow);
	}

	
}


#endif