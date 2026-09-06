# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Build & Install

```bash
python setup.py install
# or
pip install .
```

This compiles the CUDA extension `diff_gaussian_rasterization._C` using `torch.utils.cpp_extension`. Requires a CUDA-capable GPU and matching PyTorch/CUDA versions. GLM is included as a submodule in `third_party/glm/` and is passed as an include path to nvcc.

## Architecture

This is a PyTorch C++/CUDA extension that implements differentiable 3D Gaussian Splatting rasterization. It extends the original Inria/GRAPHDECO implementation with depth rendering and optical flow rendering.

### Layer stack (top to bottom)

1. **Python API** (`diff_gaussian_rasterization/__init__.py`) — exposes two main interfaces:
   - `GaussianRasterizer` — standard rasterizer; `forward()` returns `(color, radii, depth)`
   - `GaussianRasterizerWithFlow` — differentiable colour + depth + optical-flow rasterizer; `forward()` returns `(color, radii, depth, flow)`; no cache needed. Both colour and flow are differentiable — see [`docs/flow-forward-pass.md`](docs/flow-forward-pass.md) and [`docs/flow-backward-design.md`](docs/flow-backward-design.md)

2. **Custom-operator registration** (`ext.cpp`) — declares the schemas with `TORCH_LIBRARY` and binds the CUDA implementations with `TORCH_LIBRARY_IMPL`, so the entry points are reached as `torch.ops.diff_gaussian_rasterization.<name>`. The `_C` extension module only exists so that importing it loads the shared object and triggers registration; `__init__.py` keeps the old `_C.<name>` attributes as aliases for back-compat

3. **CUDA/PyTorch bridge** (`rasterize_points.cu` / `rasterize_points.h`) — allocates PyTorch tensors and calls into the rasterizer; defines `RasterizeGaussiansCUDA`, `RasterizeGaussiansBackwardCUDA`, `RasterizeGaussiansWithFlowCUDA`, `RasterizeGaussiansWithFlowBackwardCUDA`, `markVisible`

4. **Core CUDA rasterizer** (`cuda_rasterizer/`):
   - `rasterizer.h` — public interface (`CudaRasterizer::Rasterizer`, `CudaRasterizer::DiffFlowRasterizer`)
   - `rasterizer_impl.cu` / `rasterizer_impl.h` — orchestrates tile-based rendering: frustum culling, depth sorting, tile binning, kernel launch
   - `forward.cu` — CUDA kernels for preprocessing Gaussians (SH → color, project to 2D, compute 2D covariance) and the per-pixel alpha-compositing render kernels (colour, and colour+depth+flow)
   - `backward.cu` — CUDA kernels for the backward pass: colour gradients (`renderCUDA`/`preprocess`) and the differentiable-flow gradients (`diffFlowRenderBackwardCUDA` + the previous-frame preprocess). **Depth backward is still not implemented.**
   - `auxiliary.h` — math helpers (projection, covariance computation)
   - `config.h` — tile size constant (`BLOCK_X`, `BLOCK_Y`)

> The earlier **cache-based** flow path (`GaussianFlowRasterizer`, `create_cache`, `FlowRasterizer`, `FORWARD::FLOW::render`, `RasterizeGaussiansFlowCUDA`, and the cache/header state structs) has been removed; the differentiable `GaussianRasterizerWithFlow` supersedes it.

### Depth computation

Depth uses **median depth** by default: the depth of the Gaussian center whose contribution causes accumulated ray transmittance to drop below 0.5. Pixels where no Gaussian reaches the threshold receive a default depth of `15`. (An alternative mean-depth formulation is commented out in `forward.cu`.)

**The backward pass for depth is not implemented** — gradients through depth will not work for depth-supervised training.

### Data flow for standard rasterization

```
GaussianRasterizer.forward()
  → rasterize_gaussians()        # Python
  → _RasterizeGaussians.apply()  # torch.autograd.Function
  → torch.ops.diff_gaussian_rasterization.rasterize_gaussians()  # custom op
  → RasterizeGaussiansCUDA()     # rasterize_points.cu
  → CudaRasterizer::Rasterizer::forward()  # rasterizer_impl.cu
  → [preprocess kernel] + [render kernel]  # forward.cu
```

### GaussianRasterizationSettings fields

`bg`, `image_height`, `image_width`, `tanfovx`, `tanfovy`, `scale_modifier`, `viewmatrix`, `projmatrix`, `sh_degree`, `campos`, `prefiltered`, `device_id`