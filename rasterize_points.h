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

#pragma once
#include <torch/types.h>
#include <cstdio>
#include <tuple>
#include <string>
	
std::tuple<int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const double scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const double tan_fovx, 
	const double tan_fovy,
    const int64_t image_height,
    const int64_t image_width,
	const torch::Tensor& sh,
	const int64_t degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const int64_t device_id);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const double scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const double tan_fovx, 
	const double tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& sh,
	const int64_t degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int64_t R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int64_t device_id);
		
torch::Tensor markVisible(
		const torch::Tensor& means3D,
		const torch::Tensor& viewmatrix,
		const torch::Tensor& projmatrix,
		const int64_t device_id);

// Differentiable version: same weights for colour and flow, no cache needed.
std::tuple<int64_t, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithFlowCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& prev_means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& prev_scales,
	const torch::Tensor& rotations,
	const torch::Tensor& prev_rotations,
	const double scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& prev_cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const double tan_fovx,
	const double tan_fovy,
    const int64_t image_height,
    const int64_t image_width,
	const torch::Tensor& sh,
	const int64_t degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const int64_t device_id);

// Backward of the differentiable flow rasterizer. Returns current-frame
// gradients plus previous-frame (flow-specific) gradients.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansWithFlowBackwardCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& prev_means3D,
	const torch::Tensor& radii,
	const torch::Tensor& colors,
	const torch::Tensor& scales,
	const torch::Tensor& prev_scales,
	const torch::Tensor& rotations,
	const torch::Tensor& prev_rotations,
	const double scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& prev_cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const double tan_fovx,
	const double tan_fovy,
	const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_flow,
	const torch::Tensor& sh,
	const int64_t degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const torch::Tensor& flowBuffer,
	const int64_t R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const int64_t device_id);