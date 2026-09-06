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

#include <math.h>
#include <torch/types.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

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
	const int64_t device_id)
{
  const at::cuda::CUDAGuard device_guard(static_cast<at::DeviceIndex>(device_id));

  TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
              "means3D must have dimensions (num_points, 3)");
  
  const int P = means3D.size(0);
  const int H = static_cast<int>(image_height);
  const int W = static_cast<int>(image_width);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  
  torch::Device device(torch::kCUDA, static_cast<torch::DeviceIndex>(device_id));
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int64_t rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, static_cast<int>(degree), M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		opacity.contiguous().data_ptr<float>(), 
		scales.contiguous().data_ptr<float>(),
		static_cast<float>(scale_modifier),
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(), 
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		static_cast<float>(tan_fovx),
		static_cast<float>(tan_fovy),
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		out_depth.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>());
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_depth);
}

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
	const int64_t device_id) 
{
  const at::cuda::CUDAGuard device_guard(static_cast<at::DeviceIndex>(device_id));

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, static_cast<int>(degree), M, static_cast<int>(R),
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  static_cast<float>(scale_modifier),
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  static_cast<float>(tan_fovx),
	  static_cast<float>(tan_fovy),
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),  
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>());
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		const torch::Tensor& means3D,
		const torch::Tensor& viewmatrix,
		const torch::Tensor& projmatrix,
		const int64_t device_id)
{ 
  const at::cuda::CUDAGuard device_guard(static_cast<at::DeviceIndex>(device_id));

  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}

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
	const int64_t device_id)
{
  const at::cuda::CUDAGuard device_guard(static_cast<at::DeviceIndex>(device_id));

  TORCH_CHECK(means3D.ndimension() == 2 && means3D.size(1) == 3,
              "means3D must have dimensions (num_points, 3)");

  const int P = means3D.size(0);
  const int H = static_cast<int>(image_height);
  const int W = static_cast<int>(image_width);

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_flow = torch::full({2, H, W}, 0.0, float_opts);

  torch::Device device(torch::kCUDA, static_cast<torch::DeviceIndex>(device_id));
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor flowBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> flowFunc = resizeFunctional(flowBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  int64_t rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::DiffFlowRasterizer::forward(
	    geomFunc,
		flowFunc,
		binningFunc,
		imgFunc,
	    P, static_cast<int>(degree), M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		prev_means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(),
		opacity.contiguous().data_ptr<float>(),
		scales.contiguous().data_ptr<float>(),
		prev_scales.contiguous().data_ptr<float>(),
		static_cast<float>(scale_modifier),
		rotations.contiguous().data_ptr<float>(),
		prev_rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(),
		prev_cov3D_precomp.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		static_cast<float>(tan_fovx),
		static_cast<float>(tan_fovy),
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		out_depth.contiguous().data_ptr<float>(),
		out_flow.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>());
  }
  return std::make_tuple(rendered, out_color, radii, geomBuffer, flowBuffer, binningBuffer, imgBuffer, out_depth, out_flow);
}

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
	const int64_t device_id)
{
  const at::cuda::CUDAGuard device_guard(static_cast<at::DeviceIndex>(device_id));

  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);

  int M = 0;
  if(sh.size(0) != 0)
  {
	M = sh.size(1);
  }

  // Current-frame gradients (mirrors the colour backward).
  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  // Previous-frame (flow-specific) gradients.
  torch::Tensor dL_dprev_means3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dprev_cov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dprev_scales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dprev_rotations = torch::zeros({P, 4}, means3D.options());

  if(P != 0)
  {
	  CudaRasterizer::DiffFlowRasterizer::backward(P, static_cast<int>(degree), M, static_cast<int>(R),
	  background.contiguous().data_ptr<float>(),
	  W, H,
	  means3D.contiguous().data_ptr<float>(),
	  prev_means3D.contiguous().data_ptr<float>(),
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  scales.data_ptr<float>(),
	  prev_scales.data_ptr<float>(),
	  static_cast<float>(scale_modifier),
	  rotations.data_ptr<float>(),
	  prev_rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data_ptr<float>(),
	  prev_cov3D_precomp.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  static_cast<float>(tan_fovx),
	  static_cast<float>(tan_fovy),
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(flowBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_flow.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dconic.contiguous().data_ptr<float>(),
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dmeans3D.contiguous().data_ptr<float>(),
	  dL_dcov3D.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dscales.contiguous().data_ptr<float>(),
	  dL_drotations.contiguous().data_ptr<float>(),
	  dL_dprev_means3D.contiguous().data_ptr<float>(),
	  dL_dprev_cov3D.contiguous().data_ptr<float>(),
	  dL_dprev_scales.contiguous().data_ptr<float>(),
	  dL_dprev_rotations.contiguous().data_ptr<float>());
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations,
                         dL_dprev_means3D, dL_dprev_cov3D, dL_dprev_scales, dL_dprev_rotations);
}