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

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

void checkValidCUDAPointer(void *ptr) 
{
	cudaPointerAttributes attr;
	cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
	if (err != cudaSuccess) {
		printf("Invalid or unrecognized CUDA pointer: %s\n", cudaGetErrorString(err));
	} else {
		// attr.type tells you if it's device, host, or managed memory
		printf("Pointer type: %d\n", attr.type);
	}
}

#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	obtain(chunk, geom.bbx_min, P, 128);
	obtain(chunk, geom.bbx_max, P, 128);

	return geom;
}

CudaRasterizer::FlowState CudaRasterizer::FlowState::fromChunk(char*& chunk, size_t P)
{
	FlowState flow;
	obtain(chunk, flow.prev_means2D, P, 128);
	obtain(chunk, flow.prev_cov2D_opacity, P, 128);
	obtain(chunk, flow.sqrt_conic, P, 128);
	obtain(chunk, flow.prev_sqrt_cov2D, P, 128);
	return flow;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

CudaRasterizer::GaussianHeaderState CudaRasterizer::GaussianHeaderState::fromChunk(char*& chunk, size_t P)
{
	GaussianHeaderState header;
	obtain(chunk, header.bbx_min, P, 128);
	obtain(chunk, header.bbx_max, P, 128);
	obtain(chunk, header.cache_offset, P, 128);
	return header;
}

CudaRasterizer::CacheState CudaRasterizer::CacheState::fromChunk(char*& chunk, size_t N)
{
	CacheState cache;
	obtain(chunk, cache.t_value, N, 128);
	obtain(chunk, cache.g_value, N, 128);
	return cache;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	int* radii)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	int img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.bbx_min,
		geomState.bbx_max,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CUDA_CHECK(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost));

	int binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	CUDA_CHECK(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)));

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		out_depth);

	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot)
{
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot);
}

// Create a cache of transmittance and Gaussian function values per pixel.
void CudaRasterizer::Rasterizer::createCache(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	std::function<char* (size_t)> gaussianHeaderBuffer,
	std::function<char* (size_t)> cacheBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	int* radii)
{

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	int img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		geomState.bbx_min,
		geomState.bbx_max,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CUDA_CHECK(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost));

	int binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	CUDA_CHECK(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)));

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;

	// Create transmittance cache for flow rendering
	GaussianHeaderState gaussianHeaderState;
	CacheState cacheState;

	CudaRasterizer::allocateCache(
		P, tile_grid, block,
		imgState, binningState, geomState,
		gaussianHeaderBuffer, cacheBuffer,
		width, height,
		feature_ptr,
		background,
		geomState.depths,
		gaussianHeaderState,
		cacheState);

	// Let each tile blend its range of Gaussians independently in parallel
	FORWARD::renderCache(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height, P,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		geomState.depths,
		gaussianHeaderState.bbx_min,
		gaussianHeaderState.bbx_max,
		gaussianHeaderState.cache_offset,
		cacheState.t_value,
		cacheState.g_value);
}

void computeSumCUDA(uint32_t P, 
	const uint64_t* __restrict__ counts, 
	uint64_t* __restrict__ sum)
{
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	// Get the size of the temporary storage needed for the scan
	cub::DeviceReduce::Sum(
		d_temp_storage, temp_storage_bytes,
		counts, sum, P
	);

	if (temp_storage_bytes > 0)
	{
		CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	}

	// Perform the reduction
	cub::DeviceReduce::Sum(
		d_temp_storage, temp_storage_bytes,
		counts, sum, P
	);

	// Free the temporary storage
	if (d_temp_storage != nullptr)
	{
		CUDA_CHECK(cudaFree(d_temp_storage));
	}
}

void computePrefixSumCUDA(
	uint64_t* __restrict__ counts,
	uint64_t* __restrict__ offsets,
	uint32_t P)
{
	void* d_temp_storage = nullptr;
	size_t temp_storage_bytes = 0;

	// Get the size of the temporary storage needed for the scan
	cub::DeviceScan::ExclusiveSum(
		d_temp_storage, temp_storage_bytes,
		counts, offsets, P
	);

	if (temp_storage_bytes > 0)
	{
		CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
	}

	// Perform the prefix sum
	cub::DeviceScan::ExclusiveSum(
		d_temp_storage, temp_storage_bytes,
		counts, offsets, P
	);

	// Free the temporary storage
	if (d_temp_storage != nullptr)
	{
		CUDA_CHECK(cudaFree(d_temp_storage));
	}
}

void CudaRasterizer::allocateCache(
	int P, dim3 tile_grid, dim3 block,
	const ImageState& imgState,
	const BinningState& binningState,
	const GeometryState& geomState,
	std::function<char* (size_t)> gaussianHeaderBuffer,
	std::function<char* (size_t)> cacheBuffer,
	const int width, const int height,
	const float* feature_ptr,
	const float* bg_color,
	const float* depth,
	GaussianHeaderState& out_gaussianHeaderState,
	CacheState& out_cacheState	
)
{
	// Allocate temporary variables for computing the buffer sizes dynamically
	uint64_t* cache_counts_per_gaussian; // Number of T, G values per Gaussian
	uint64_t* cache_offsets; // Offsets computed from cache counts

	CUDA_CHECK(cudaMalloc((void**)&cache_counts_per_gaussian, P * sizeof(uint64_t)));
	CUDA_CHECK(cudaMalloc((void**)&cache_offsets, P * sizeof(uint64_t)));

	printf("Test1\n");

	FORWARD::computeCacheLayout(
		P, width, height,
		geomState.bbx_min,
		geomState.bbx_max,
		cache_counts_per_gaussian
	);

	printf("Test2\n");

	// Compute the total number of cache entries needed
	uint64_t total_cache_count = 0;
	uint64_t* d_total_cache_count;
	CUDA_CHECK(cudaMalloc((void**)&d_total_cache_count, sizeof(uint64_t)));
	computeSumCUDA(P, cache_counts_per_gaussian, d_total_cache_count);
	CUDA_CHECK(cudaMemcpy(&total_cache_count, d_total_cache_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_total_cache_count));

	printf("Test3\n");

	// Compute prefix sum over gaussians
	computePrefixSumCUDA(cache_counts_per_gaussian, cache_offsets, P);

	printf("Test4\n");

	// Allocate data structure
	size_t gaussian_header_chunk_size = required<GaussianHeaderState>(P);
	char* gaussian_header_chunkptr = gaussianHeaderBuffer(gaussian_header_chunk_size);
	out_gaussianHeaderState = GaussianHeaderState::fromChunk(gaussian_header_chunkptr, P);

	size_t cache_chunk_size = required<CacheState>(total_cache_count);
	char* cache_chunkptr = cacheBuffer(cache_chunk_size);
	out_cacheState = CacheState::fromChunk(cache_chunkptr, total_cache_count);

	printf("Test5\n");

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	CUDA_CHECK(cudaMemcpyAsync(
		out_gaussianHeaderState.bbx_min, geomState.bbx_min,
		P * sizeof(uint2), cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(
		out_gaussianHeaderState.bbx_max, geomState.bbx_max,
		P * sizeof(uint2), cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemcpyAsync(
		out_gaussianHeaderState.cache_offset, cache_offsets,
		P * sizeof(uint64_t), cudaMemcpyDeviceToDevice, stream));
	CUDA_CHECK(cudaMemsetAsync(out_cacheState.t_value, 0, total_cache_count * sizeof(float), stream));
	CUDA_CHECK(cudaMemsetAsync(out_cacheState.g_value, 0, total_cache_count * sizeof(float), stream));

	CUDA_CHECK(cudaStreamSynchronize(stream));
	CUDA_CHECK(cudaStreamDestroy(stream));

	// Free temporary variables
	CUDA_CHECK(cudaFree(cache_counts_per_gaussian));
	CUDA_CHECK(cudaFree(cache_offsets));

	CUDA_CHECK(cudaDeviceSynchronize());

	uint2* h_bbx_min;
	uint2* h_bbx_max;
	uint64_t *h_cache_offsets;
	CUDA_CHECK(cudaMallocHost((void**)&h_cache_offsets, P * sizeof(uint64_t)));
	CUDA_CHECK(cudaMallocHost((void**)&h_bbx_min, P * sizeof(uint2)));
	CUDA_CHECK(cudaMallocHost((void**)&h_bbx_max, P * sizeof(uint2)));
	CUDA_CHECK(cudaMemcpy(h_cache_offsets, out_gaussianHeaderState.cache_offset, P * sizeof(uint64_t), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_bbx_min, out_gaussianHeaderState.bbx_min, P * sizeof(uint2), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(h_bbx_max, out_gaussianHeaderState.bbx_max, P * sizeof(uint2), cudaMemcpyDeviceToHost));

	printf("---------------------------------------------\n");
	printf("Cache layout for %d Gaussians:\n", P);
	printf("Total cache entries: %u\n", total_cache_count);
	printf("Cache size: %zu bytes\n", cache_chunk_size);
	printf("---------------------------------------------\n");

	// Iterate over all Gaussian headers
	for (int i = 0; i < 1000; i++)
	{
		// Compute the bounding box for the Gaussian
		uint2 bbx_min = h_bbx_min[i];
		uint2 bbx_max = h_bbx_max[i];	

		// Compute the cache offset for this Gaussian
		uint32_t cache_offset = h_cache_offsets[i];

		// Print the bounding box and cache offset
		printf("Gaussian %d: BBX Min: (%u, %u), BBX Max: (%u, %u), Cache Offset: %u\n",
			i, bbx_min.x, bbx_min.y,
			bbx_max.x, bbx_max.y, cache_offset);
	}

	CUDA_CHECK(cudaFreeHost(h_bbx_min));
	CUDA_CHECK(cudaFreeHost(h_bbx_max));
	CUDA_CHECK(cudaFreeHost(h_cache_offsets));
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
int CudaRasterizer::FlowRasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> flowBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* prev_means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* prev_opacities,
	const float* scales,
	const float* prev_scales,
	const float scale_modifier,
	const float* rotations,
	const float* prev_rotations,
	const float* cov3D_precomp,
	const float* prev_cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	char* gaussianHeaderBuffer,
	const uint32_t gaussianHeaderBufferSize,
	char* cacheBuffer,
	const uint32_t cacheBufferSize,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	float* out_depth,
	float* out_flow,
	int* radii)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	size_t flow_chunk_size = required<FlowState>(P);
	char* flow_chunkptr = flowBuffer(flow_chunk_size);
	FlowState flowState = FlowState::fromChunk(flow_chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	int img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	FORWARD::FLOW::preprocess(
		P, D, M,
		means3D,
		prev_means3D,
		(glm::vec3*)scales,
		(glm::vec3*)prev_scales,
		scale_modifier,
		(glm::vec4*)rotations,
		(glm::vec4*)prev_rotations,
		opacities,
		prev_opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		prev_cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		flowState.prev_means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		flowState.prev_cov2D_opacity,
		flowState.sqrt_conic,
		flowState.prev_sqrt_cov2D,
		prefiltered
	);

	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	CUDA_CHECK(cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost));

	int binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);

	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	CUDA_CHECK(cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2)));

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);


	// Cast the cache buffers to the appropriate types
	GaussianHeaderState gaussianHeaderState = GaussianHeaderState::fromChunk(
		gaussianHeaderBuffer, gaussianHeaderBufferSize);
	CacheState cacheState = CacheState::fromChunk(
		cacheBuffer, cacheBufferSize);

	CUDA_CHECK(cudaDeviceSynchronize());

	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	FORWARD::FLOW::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		P, width, height,
		geomState.means2D,
		flowState.prev_means2D,
		feature_ptr,
		geomState.conic_opacity,
		flowState.prev_cov2D_opacity,
		flowState.sqrt_conic,
		flowState.prev_sqrt_cov2D,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color,
		geomState.depths,
		out_depth,
		gaussianHeaderState.bbx_min,
		gaussianHeaderState.bbx_max,
		gaussianHeaderState.cache_offset,
		cacheState.t_value,
		cacheState.g_value,
		out_flow);

	CUDA_CHECK(cudaDeviceSynchronize());

	return num_rendered;
}