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

#include <Python.h>
#include <torch/library.h>
#include "rasterize_points.h"

// The rasterizer entry points are registered as PyTorch custom operators in
// the `diff_gaussian_rasterization` namespace, i.e. they are reached from
// Python as `torch.ops.diff_gaussian_rasterization.<name>`. Schema scalars are
// `int` (int64_t) and `float` (double); rasterize_points.h matches those types.

TORCH_LIBRARY(diff_gaussian_rasterization, m) {
  m.def(
      "rasterize_gaussians("
      "Tensor background, Tensor means3D, Tensor colors, Tensor opacity, "
      "Tensor scales, Tensor rotations, float scale_modifier, "
      "Tensor cov3D_precomp, Tensor viewmatrix, Tensor projmatrix, "
      "float tan_fovx, float tan_fovy, int image_height, int image_width, "
      "Tensor sh, int degree, Tensor campos, bool prefiltered, int device_id) "
      "-> (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def(
      "rasterize_gaussians_backward("
      "Tensor background, Tensor means3D, Tensor radii, Tensor colors, "
      "Tensor scales, Tensor rotations, float scale_modifier, "
      "Tensor cov3D_precomp, Tensor viewmatrix, Tensor projmatrix, "
      "float tan_fovx, float tan_fovy, Tensor dL_dout_color, Tensor sh, "
      "int degree, Tensor campos, Tensor geomBuffer, int R, "
      "Tensor binningBuffer, Tensor imageBuffer, int device_id) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def(
      "mark_visible("
      "Tensor means3D, Tensor viewmatrix, Tensor projmatrix, int device_id) "
      "-> Tensor");

  m.def(
      "rasterize_gaussians_with_flow("
      "Tensor background, Tensor means3D, Tensor prev_means3D, Tensor colors, "
      "Tensor opacity, Tensor scales, Tensor prev_scales, Tensor rotations, "
      "Tensor prev_rotations, float scale_modifier, Tensor cov3D_precomp, "
      "Tensor prev_cov3D_precomp, Tensor viewmatrix, Tensor projmatrix, "
      "float tan_fovx, float tan_fovy, int image_height, int image_width, "
      "Tensor sh, int degree, Tensor campos, bool prefiltered, int device_id) "
      "-> (int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)");

  m.def(
      "rasterize_gaussians_with_flow_backward("
      "Tensor background, Tensor means3D, Tensor prev_means3D, Tensor radii, "
      "Tensor colors, Tensor scales, Tensor prev_scales, Tensor rotations, "
      "Tensor prev_rotations, float scale_modifier, Tensor cov3D_precomp, "
      "Tensor prev_cov3D_precomp, Tensor viewmatrix, Tensor projmatrix, "
      "float tan_fovx, float tan_fovy, Tensor dL_dout_color, "
      "Tensor dL_dout_flow, Tensor sh, int degree, Tensor campos, "
      "Tensor geomBuffer, Tensor flowBuffer, int R, Tensor binningBuffer, "
      "Tensor imageBuffer, int device_id) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
      "Tensor, Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(diff_gaussian_rasterization, CUDA, m) {
  m.impl("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.impl("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.impl("mark_visible", &markVisible);
  m.impl("rasterize_gaussians_with_flow", &RasterizeGaussiansWithFlowCUDA);
  m.impl("rasterize_gaussians_with_flow_backward", &RasterizeGaussiansWithFlowBackwardCUDA);
}

// `mark_visible` is the only entry point with a statically known output shape,
// so it is the only one that can be traced without running the kernel. The
// rasterizer ops allocate buffers whose sizes depend on the rendered tile/point
// counts, so they stay data-dependent and cause a graph break under
// torch.compile rather than being given an incorrect fake implementation.
namespace {
torch::Tensor markVisibleMeta(
    const torch::Tensor& means3D,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const int64_t device_id) {
  return torch::empty({means3D.size(0)}, means3D.options().dtype(torch::kBool));
}
}  // namespace

TORCH_LIBRARY_IMPL(diff_gaussian_rasterization, Meta, m) {
  m.impl("mark_visible", &markVisibleMeta);
}

extern "C" {
// The operators above register themselves when this shared object is loaded;
// this stub only exists so that `from . import _C` can perform that load.
// There are no pybind11 bindings any more -- call the ops through
// `torch.ops.diff_gaussian_rasterization`.
PyObject* PyInit__C(void) {
  static struct PyModuleDef module_def = {
      PyModuleDef_HEAD_INIT, "_C", nullptr, -1, nullptr};
  return PyModule_Create(&module_def);
}
}
