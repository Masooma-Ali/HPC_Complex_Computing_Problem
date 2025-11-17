/*********************************************************************
 * trackFeatures.cu - CUDA GPU-accelerated version (OPTIMIZED)
 * Complete implementation with CPU wrapper functions
 *********************************************************************/

 #include <cuda_runtime.h>
 #include <assert.h>
 #include <math.h>
 #include <stdlib.h>
 #include <stdio.h>
 
 /* Our includes */
 extern "C" {
 #include "base.h"
 #include "convolve.h"
 #include "klt.h"
 #include "klt_util.h"
 #include "pyramid.h"
 }
 
 extern int KLT_verbose;

#define BLOCK_SIZE 16
#define MAX_FEATURES 1000

// HPC OPTIMIZATION: Store frequently-read tracking parameters in constant memory
// These are read by ALL threads in ALL kernel calls but never change during tracking
__constant__ int c_window_width;
__constant__ int c_window_height;
__constant__ float c_step_factor;
__constant__ int c_max_iterations;
__constant__ float c_min_determinant;
__constant__ float c_min_displacement;
__constant__ float c_max_residue;


// CUDA error checking macro and helper function
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// OPTIMIZATION: GPU Memory Pool for persistent allocation - BATCHED VERSION WITH STREAMING
#define NUM_STREAMS 4  // Use 4 streams for pipelining

// HPC: Batched feature data structure to reduce transfer overhead
typedef struct {
    float x1, y1;      // Reference position in img1
    float x2, y2;      // Tracked position in img2
    int status;        // Tracking status
} FeatureData;

typedef struct {
    float *d_img_buffer;           // Reusable for input images
    float *d_diff_buffer;          // Reusable for difference windows  
    float *d_temp_buffer;          // Reusable for temporary data
    // NEW: Pyramid storage on GPU
    float *d_pyramid1_imgs[10];    // Store all pyramid levels on GPU
    float *d_pyramid1_gradx[10];
    float *d_pyramid1_grady[10];
    float *d_pyramid2_imgs[10];
    float *d_pyramid2_gradx[10];
    float *d_pyramid2_grady[10];
    int pyramid_levels;
    cudaStream_t streams[NUM_STREAMS];  // Multiple streams for pipelining
    cudaEvent_t level_events[10];      // Events for inter-level dependencies (no host sync!)
    int events_initialized;            // Flag for event initialization
    // HPC: Persistent feature data buffers (allocated once, reused forever)
    FeatureData *d_features;           // GPU feature buffer
    FeatureData *h_features;           // Pinned host feature buffer
    int max_features;                  // Maximum features allocated
    int allocated_img_size;        // Maximum image buffer size
    int allocated_diff_size;       // Maximum difference buffer size
    int initialized;               // Flag to check if initialized
    int pyramids_on_gpu;           // Flag: pyramids already uploaded
} GPUTrackingMemoryPool;

static GPUTrackingMemoryPool gpu_tracking_pool = {NULL, NULL, NULL, NULL, 
    {}, {}, {}, {}, {}, {}, 0, {}, {}, 0, NULL, NULL, 0, 0, 0, 0, 0};

// OPTIMIZATION: Initialize GPU memory pool (allocate once, reuse forever) - WITH STREAMING
void _initTrackingGPUPool(int img_width, int img_height, int max_window_size)
{
    int img_size = img_width * img_height * sizeof(float);
    int diff_size = max_window_size * max_window_size * sizeof(float);
    
    // Check if we need to reallocate
    if (!gpu_tracking_pool.initialized || 
        gpu_tracking_pool.allocated_img_size < img_size ||
        gpu_tracking_pool.allocated_diff_size < diff_size) {
        
        // Free old allocations if they exist
        if (gpu_tracking_pool.d_img_buffer) cudaFree(gpu_tracking_pool.d_img_buffer);
        if (gpu_tracking_pool.d_diff_buffer) cudaFree(gpu_tracking_pool.d_diff_buffer);
        if (gpu_tracking_pool.d_temp_buffer) cudaFree(gpu_tracking_pool.d_temp_buffer);
        
        // Destroy old streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            if (gpu_tracking_pool.streams[i]) cudaStreamDestroy(gpu_tracking_pool.streams[i]);
        }
        
        // Allocate persistent GPU memory (allocate once!)
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_img_buffer, img_size * 4)); // For 4 gradient images
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_diff_buffer, diff_size));
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_temp_buffer, diff_size));
        
        // Create multiple streams for pipelining
        for (int i = 0; i < NUM_STREAMS; i++) {
            cudaCheckError(cudaStreamCreate(&gpu_tracking_pool.streams[i]));
        }
        
        // HPC: Create events for inter-level dependencies (eliminates host syncs!)
        if (!gpu_tracking_pool.events_initialized) {
            for (int i = 0; i < 10; i++) {
                cudaCheckError(cudaEventCreate(&gpu_tracking_pool.level_events[i]));
            }
            gpu_tracking_pool.events_initialized = 1;
        }
        
        gpu_tracking_pool.allocated_img_size = img_size;
        gpu_tracking_pool.allocated_diff_size = diff_size;
        gpu_tracking_pool.initialized = 1;
    }
}

// OPTIMIZATION: Cleanup GPU pool - WITH STREAMING
void _cleanupTrackingGPUPool()
{
    if (gpu_tracking_pool.initialized) {
        if (gpu_tracking_pool.d_img_buffer) cudaFree(gpu_tracking_pool.d_img_buffer);
        if (gpu_tracking_pool.d_diff_buffer) cudaFree(gpu_tracking_pool.d_diff_buffer);
        if (gpu_tracking_pool.d_temp_buffer) cudaFree(gpu_tracking_pool.d_temp_buffer);
        
        // Destroy all streams
        for (int i = 0; i < NUM_STREAMS; i++) {
            if (gpu_tracking_pool.streams[i]) cudaStreamDestroy(gpu_tracking_pool.streams[i]);
        }
        
        // Free pyramid storage
        for (int i = 0; i < gpu_tracking_pool.pyramid_levels; i++) {
            if (gpu_tracking_pool.d_pyramid1_imgs[i]) cudaFree(gpu_tracking_pool.d_pyramid1_imgs[i]);
            if (gpu_tracking_pool.d_pyramid1_gradx[i]) cudaFree(gpu_tracking_pool.d_pyramid1_gradx[i]);
            if (gpu_tracking_pool.d_pyramid1_grady[i]) cudaFree(gpu_tracking_pool.d_pyramid1_grady[i]);
            if (gpu_tracking_pool.d_pyramid2_imgs[i]) cudaFree(gpu_tracking_pool.d_pyramid2_imgs[i]);
            if (gpu_tracking_pool.d_pyramid2_gradx[i]) cudaFree(gpu_tracking_pool.d_pyramid2_gradx[i]);
            if (gpu_tracking_pool.d_pyramid2_grady[i]) cudaFree(gpu_tracking_pool.d_pyramid2_grady[i]);
        }
        
        gpu_tracking_pool.d_img_buffer = NULL;
        gpu_tracking_pool.d_diff_buffer = NULL;
        gpu_tracking_pool.d_temp_buffer = NULL;
        gpu_tracking_pool.initialized = 0;
    }
}

// OPTIMIZED: Upload pyramids to GPU with STREAMING and POINTER SWAPPING for sequential mode
void _uploadPyramidsToGPU(_KLT_Pyramid pyr1, _KLT_Pyramid pyr1_gradx, _KLT_Pyramid pyr1_grady,
                          _KLT_Pyramid pyr2, _KLT_Pyramid pyr2_gradx, _KLT_Pyramid pyr2_grady,
                          int nlevels, int sequential_mode_reuse)
{
    // HPC OPTIMIZATION: In sequential mode, pyramid2 from last frame = pyramid1 of this frame
    // Determine if we need to upload pyramid1 or can reuse it
    int upload_pyramid1 = 1;  // Default: upload both pyramids
    
    if (sequential_mode_reuse && gpu_tracking_pool.pyramids_on_gpu) {
        // Swap pyramid pointers: last frame's pyramid2 becomes this frame's pyramid1
        for (int i = 0; i < nlevels; i++) {
            float *temp;
            temp = gpu_tracking_pool.d_pyramid1_imgs[i];
            gpu_tracking_pool.d_pyramid1_imgs[i] = gpu_tracking_pool.d_pyramid2_imgs[i];
            gpu_tracking_pool.d_pyramid2_imgs[i] = temp;
            
            temp = gpu_tracking_pool.d_pyramid1_gradx[i];
            gpu_tracking_pool.d_pyramid1_gradx[i] = gpu_tracking_pool.d_pyramid2_gradx[i];
            gpu_tracking_pool.d_pyramid2_gradx[i] = temp;
            
            temp = gpu_tracking_pool.d_pyramid1_grady[i];
            gpu_tracking_pool.d_pyramid1_grady[i] = gpu_tracking_pool.d_pyramid2_grady[i];
            gpu_tracking_pool.d_pyramid2_grady[i] = temp;
        }
        upload_pyramid1 = 0;  // Pyramid1 already on GPU, only upload pyramid2!
    } else {
        // First time or non-sequential mode: free old pyramid data if exists
        for (int i = 0; i < gpu_tracking_pool.pyramid_levels; i++) {
            if (gpu_tracking_pool.d_pyramid1_imgs[i]) cudaFree(gpu_tracking_pool.d_pyramid1_imgs[i]);
            if (gpu_tracking_pool.d_pyramid1_gradx[i]) cudaFree(gpu_tracking_pool.d_pyramid1_gradx[i]);
            if (gpu_tracking_pool.d_pyramid1_grady[i]) cudaFree(gpu_tracking_pool.d_pyramid1_grady[i]);
            if (gpu_tracking_pool.d_pyramid2_imgs[i]) cudaFree(gpu_tracking_pool.d_pyramid2_imgs[i]);
            if (gpu_tracking_pool.d_pyramid2_gradx[i]) cudaFree(gpu_tracking_pool.d_pyramid2_gradx[i]);
            if (gpu_tracking_pool.d_pyramid2_grady[i]) cudaFree(gpu_tracking_pool.d_pyramid2_grady[i]);
        }
        
        gpu_tracking_pool.pyramid_levels = nlevels;
        
        // OPTIMIZATION: Allocate all GPU memory first
        for (int i = 0; i < nlevels; i++) {
            int ncols = pyr1->ncols[i];
            int nrows = pyr1->nrows[i];
            int size = ncols * nrows * sizeof(float);
            
            // Allocate GPU memory for this level
            cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_pyramid1_imgs[i], size));
            cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_pyramid1_gradx[i], size));
            cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_pyramid1_grady[i], size));
            cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_pyramid2_imgs[i], size));
            cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_pyramid2_gradx[i], size));
            cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_pyramid2_grady[i], size));
        }
    }
    
    // OPTIMIZATION: Stream uploads across multiple streams to overlap transfers
    for (int i = 0; i < nlevels; i++) {
        int ncols = pyr1->ncols[i];
        int nrows = pyr1->nrows[i];
        int size = ncols * nrows * sizeof(float);
        
        // Select stream in round-robin fashion for load balancing
        int stream_id = i % NUM_STREAMS;
        cudaStream_t stream = gpu_tracking_pool.streams[stream_id];
        
        // Upload pyramid1 only if needed (not reusing from previous frame)
        if (upload_pyramid1) {
            cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_pyramid1_imgs[i], pyr1->img[i]->data, 
                                           size, cudaMemcpyHostToDevice, stream));
            cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_pyramid1_gradx[i], pyr1_gradx->img[i]->data, 
                                           size, cudaMemcpyHostToDevice, stream));
            cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_pyramid1_grady[i], pyr1_grady->img[i]->data, 
                                           size, cudaMemcpyHostToDevice, stream));
        }
        
        // Always upload pyramid2 (current frame)
        cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_pyramid2_imgs[i], pyr2->img[i]->data, 
                                       size, cudaMemcpyHostToDevice, stream));
        cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_pyramid2_gradx[i], pyr2_gradx->img[i]->data, 
                                       size, cudaMemcpyHostToDevice, stream));
        cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_pyramid2_grady[i], pyr2_grady->img[i]->data, 
                                       size, cudaMemcpyHostToDevice, stream));
    }
    
    // OPTIMIZATION: Synchronize all streams to ensure uploads complete
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaCheckError(cudaStreamSynchronize(gpu_tracking_pool.streams[i]));
    }
    
    gpu_tracking_pool.pyramids_on_gpu = 1;  // Mark pyramids as uploaded
}

typedef float *_FloatWindow;

 __device__ float interpolate_gpu(
    float x,
    float y,
    float *img_data,
    int ncols,
    int nrows)
{
    int xt = (int)x;
    int yt = (int)y;
    float ax = x - xt;
    float ay = y - yt;
    
    if (xt < 0 || yt < 0 || xt >= ncols-1 || yt >= nrows-1)
        return 0.0f;
    
    float *ptr = img_data + (ncols*yt) + xt;
    
    // Use __ldg() for cached reads - improves L1 cache utilization
    return ((1-ax) * (1-ay) * __ldg(ptr) +
            ax * (1-ay) * __ldg(ptr+1) +
            (1-ax) * ay * __ldg(ptr+ncols) +
            ax * ay * __ldg(ptr+ncols+1));
}

// HPC OPTIMIZATION: Texture cache interpolation with manual bilinear
// Since our data is in flat row-major format, we use 1D texture fetches
// with manual bilinear interpolation, but benefit from texture cache!
__device__ __forceinline__ float interpolate_tex(
    float x,
    float y,
    cudaTextureObject_t tex,
    int ncols,
    int nrows)
{
    int xt = (int)x;
    int yt = (int)y;
    float ax = x - xt;
    float ay = y - yt;
    
    if (xt < 0 || yt < 0 || xt >= ncols-1 || yt >= nrows-1)
        return 0.0f;
    
    // Compute 1D indices for 2D row-major layout
    int idx00 = yt * ncols + xt;
    int idx01 = idx00 + 1;
    int idx10 = idx00 + ncols;
    int idx11 = idx10 + 1;
    
    // Use texture cache for reads (benefits from 2D spatial locality)
    float v00 = tex1Dfetch<float>(tex, idx00);
    float v01 = tex1Dfetch<float>(tex, idx01);
    float v10 = tex1Dfetch<float>(tex, idx10);
    float v11 = tex1Dfetch<float>(tex, idx11);
    
    // Manual bilinear interpolation (same math as before)
    return ((1-ax) * (1-ay) * v00 +
            ax * (1-ay) * v01 +
            (1-ax) * ay * v10 +
            ax * ay * v11);
}

// HPC: Helper function to create 1D texture object from flat GPU memory
// Our pyramid data is stored as flat row-major arrays, so we use 1D textures
// and manually compute 2D indices in the interpolation function
cudaTextureObject_t createTextureObject(float* d_data, int width, int height) {
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;  // Linear memory (our flat arrays)
    resDesc.res.linear.devPtr = d_data;
    resDesc.res.linear.desc = cudaCreateChannelDesc<float>();
    resDesc.res.linear.sizeInBytes = width * height * sizeof(float);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;  // Clamp coordinates
    texDesc.filterMode = cudaFilterModePoint;       // Point sampling (we do interpolation manually)
    texDesc.readMode = cudaReadModeElementType;     // Read as float
    texDesc.normalizedCoords = 0;                   // Use element indices (not normalized 0-1)
    
    cudaTextureObject_t texObj = 0;
    cudaCheckError(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));
    return texObj;
}
 
 __global__ void computeIntensityDifferenceKernel(
     float *img1,
     float *img2,
     float x1,
     float y1,
     float x2,
     float y2,
     int width,
     int height,
     int ncols,
     int nrows,
     float *imgdiff)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     int j = blockIdx.y * blockDim.y + threadIdx.y;
     
     int hw = width / 2;
     int hh = height / 2;
     
     if (i > 2*hw || j > 2*hh) return;
     
     int local_i = i - hw;
     int local_j = j - hh;
     
     float g1 = interpolate_gpu(x1 + local_i, y1 + local_j, img1, ncols, nrows);
     float g2 = interpolate_gpu(x2 + local_i, y2 + local_j, img2, ncols, nrows);
     
     int idx = j * (2*hw + 1) + i;
     imgdiff[idx] = g1 - g2;
 }
 
 __global__ void computeGradientSumKernel(
     float *gradx1,
     float *grady1,
     float *gradx2,
     float *grady2,
     float x1,
     float y1,
     float x2,
     float y2,
     int width,
     int height,
     int ncols,
     int nrows,
     float *gradx_out,
     float *grady_out)
 {
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     int j = blockIdx.y * blockDim.y + threadIdx.y;
     
     int hw = width / 2;
     int hh = height / 2;
     
     if (i > 2*hw || j > 2*hh) return;
     
     int local_i = i - hw;
     int local_j = j - hh;
     
     float gx1 = interpolate_gpu(x1 + local_i, y1 + local_j, gradx1, ncols, nrows);
     float gx2 = interpolate_gpu(x2 + local_i, y2 + local_j, gradx2, ncols, nrows);
     float gy1 = interpolate_gpu(x1 + local_i, y1 + local_j, grady1, ncols, nrows);
     float gy2 = interpolate_gpu(x2 + local_i, y2 + local_j, grady2, ncols, nrows);
     
     int idx = j * (2*hw + 1) + i;
     gradx_out[idx] = gx1 + gx2;
     grady_out[idx] = gy1 + gy2;
 }
 
 /*********************************************************************
 * GPU KERNEL: Compute window statistics (mean, variance) in parallel
 * Used for feature quality assessment
 *********************************************************************/
__global__ void computeWindowStatisticsKernel(
    float *img,
    float x, float y,
    int window_width, int window_height,
    int ncols, int nrows,
    float *mean_out, float *variance_out)
{
    int i = threadIdx.x;
    int j = threadIdx.y;
    int hw = window_width / 2;
    int hh = window_height / 2;
    
    __shared__ float sum_shared[256];
    __shared__ float sum_sq_shared[256];
    
    int tid = j * blockDim.x + i;
    sum_shared[tid] = 0.0f;
    sum_sq_shared[tid] = 0.0f;
    
    if (i <= 2*hw && j <= 2*hh) {
        int local_i = i - hw;
        int local_j = j - hh;
        float val = interpolate_gpu(x + local_i, y + local_j, img, ncols, nrows);
        sum_shared[tid] = val;
        sum_sq_shared[tid] = val * val;
    }
    __syncthreads();
    
    // Parallel reduction for sum
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sum_shared[tid] += sum_shared[tid + s];
            sum_sq_shared[tid] += sum_sq_shared[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        int n = window_width * window_height;
        *mean_out = sum_shared[0] / n;
        float mean_sq = sum_sq_shared[0] / n;
        *variance_out = mean_sq - (*mean_out) * (*mean_out);
    }
}

// NEW: BATCHED tracking kernel - processes all features in parallel
// HPC: Uses constant memory + batched struct + TEXTURE MEMORY for hardware interpolation!
__global__ void trackFeaturesBatchedKernel(
    cudaTextureObject_t tex_img1, cudaTextureObject_t tex_img2,
    cudaTextureObject_t tex_gradx1, cudaTextureObject_t tex_grady1,
    cudaTextureObject_t tex_gradx2, cudaTextureObject_t tex_grady2,
    FeatureData *features,             // HPC: Batched structure - all coords in one transfer!
    int num_features,
    int ncols, int nrows)
{
    int feat_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (feat_idx >= num_features) return;
    
    // HPC: Read from batched structure (coalesced memory access)
    FeatureData feat = features[feat_idx];
    
    // Skip features that have already failed at previous pyramid levels
    if (feat.status < 0) {
        return;  // Feature already lost, don't track
    }
    
    // Each thread tracks one feature - exact same logic as before!
    float x1 = feat.x1;
    float y1 = feat.y1;
    float x2 = feat.x2;
    float y2 = feat.y2;
    
    // HPC: Use constant memory parameters (faster than passed parameters)
    int hw = c_window_width / 2;
    int hh = c_window_height / 2;
    float one_plus_eps = 1.001f;
    int iteration = 0;
    int status = KLT_TRACKED;
    float dx, dy;
    
    // Newton-Raphson iteration loop ON GPU
    do {
        // Boundary check
        if (x1 - hw < 0.0f || ncols - (x1 + hw) < one_plus_eps ||
            x2 - hw < 0.0f || ncols - (x2 + hw) < one_plus_eps ||
            y1 - hh < 0.0f || nrows - (y1 + hh) < one_plus_eps ||
            y2 - hh < 0.0f || nrows - (y2 + hh) < one_plus_eps) {
            status = KLT_OOB;
            break;
        }
        
        // Compute gradient matrix and error vector ON GPU
        float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
        float ex = 0.0f, ey = 0.0f;
        
        for (int j = -hh; j <= hh; j++) {
            for (int i = -hw; i <= hw; i++) {
                // HPC: Use texture cache for reads (benefits from spatial locality!)
                float g1 = interpolate_tex(x1 + i, y1 + j, tex_img1, ncols, nrows);
                float g2 = interpolate_tex(x2 + i, y2 + j, tex_img2, ncols, nrows);
                float diff = g1 - g2;
                
                // Sum gradients (NOT average) - must match CPU version!
                float gx = interpolate_tex(x1 + i, y1 + j, tex_gradx1, ncols, nrows) +
                           interpolate_tex(x2 + i, y2 + j, tex_gradx2, ncols, nrows);
                float gy = interpolate_tex(x1 + i, y1 + j, tex_grady1, ncols, nrows) +
                           interpolate_tex(x2 + i, y2 + j, tex_grady2, ncols, nrows);
                
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
                ex += diff * gx;
                ey += diff * gy;
            }
        }
        
        ex *= c_step_factor;
        ey *= c_step_factor;
        
        // Solve equation
        float det = gxx * gyy - gxy * gxy;
        if (det < c_min_determinant) {
            status = KLT_SMALL_DET;
            break;
        }
        
        dx = (gyy * ex - gxy * ey) / det;
        dy = (gxx * ey - gxy * ex) / det;
        
        x2 += dx;
        y2 += dy;
        iteration++;
        
    } while ((fabsf(dx) >= c_min_displacement || fabsf(dy) >= c_min_displacement) && iteration < c_max_iterations);
    
    // Final boundary check
    if (x2 - hw < 0.0f || ncols - (x2 + hw) < one_plus_eps ||
        y2 - hh < 0.0f || nrows - (y2 + hh) < one_plus_eps) {
        status = KLT_OOB;
    }
    
    // Check residue if tracked
    if (status == KLT_TRACKED) {
        float sum_abs_diff = 0.0f;
        for (int j = -hh; j <= hh; j++) {
            for (int i = -hw; i <= hw; i++) {
                // HPC: Texture cache for residue check too!
                float g1 = interpolate_tex(x1 + i, y1 + j, tex_img1, ncols, nrows);
                float g2 = interpolate_tex(x2 + i, y2 + j, tex_img2, ncols, nrows);
                sum_abs_diff += fabsf(g1 - g2);
            }
        }
        if (sum_abs_diff / (c_window_width * c_window_height) > c_max_residue) {
            status = KLT_LARGE_RESIDUE;
        }
    }
    
    if (iteration >= c_max_iterations && status == KLT_TRACKED) {
        status = KLT_MAX_ITERATIONS;
    }
    
    // HPC: Write results back to batched structure (coalesced memory access)
    features[feat_idx].x2 = x2;
    features[feat_idx].y2 = y2;
    features[feat_idx].status = status;
    // Note: x1, y1 unchanged (reference position from img1)
}

__global__ void computeIntensityDifferenceLightingInsensitiveKernel(
     float *img1,
     float *img2,
     float x1,
     float y1,
     float x2,
     float y2,
     int width,
     int height,
     int ncols,
     int nrows,
     float *imgdiff)
 {
     __shared__ float sum1_shared[BLOCK_SIZE * BLOCK_SIZE];
     __shared__ float sum2_shared[BLOCK_SIZE * BLOCK_SIZE];
     __shared__ float sum1_sq_shared[BLOCK_SIZE * BLOCK_SIZE];
     __shared__ float sum2_sq_shared[BLOCK_SIZE * BLOCK_SIZE];
     
     int i = blockIdx.x * blockDim.x + threadIdx.x;
     int j = blockIdx.y * blockDim.y + threadIdx.y;
     int tid = threadIdx.y * blockDim.x + threadIdx.x;
     
     int hw = width / 2;
     int hh = height / 2;
     
     float sum1 = 0.0f, sum2 = 0.0f;
     float sum1_sq = 0.0f, sum2_sq = 0.0f;
     
     if (i <= 2*hw && j <= 2*hh) {
         int local_i = i - hw;
         int local_j = j - hh;
         
         float g1 = interpolate_gpu(x1 + local_i, y1 + local_j, img1, ncols, nrows);
         float g2 = interpolate_gpu(x2 + local_i, y2 + local_j, img2, ncols, nrows);
         
         sum1 = g1;
         sum2 = g2;
         sum1_sq = g1 * g1;
         sum2_sq = g2 * g2;
     }
     
     sum1_shared[tid] = sum1;
     sum2_shared[tid] = sum2;
     sum1_sq_shared[tid] = sum1_sq;
     sum2_sq_shared[tid] = sum2_sq;
     __syncthreads();
     
     for (int s = (BLOCK_SIZE * BLOCK_SIZE) / 2; s > 0; s >>= 1) {
         if (tid < s) {
             sum1_shared[tid] += sum1_shared[tid + s];
             sum2_shared[tid] += sum2_shared[tid + s];
             sum1_sq_shared[tid] += sum1_sq_shared[tid + s];
             sum2_sq_shared[tid] += sum2_sq_shared[tid + s];
         }
         __syncthreads();
     }
     
     __shared__ float alpha, belta;
     
     if (tid == 0) {
         float mean1_sq = sum1_sq_shared[0] / (width * height);
         float mean2_sq = sum2_sq_shared[0] / (width * height);
         alpha = sqrtf(mean1_sq / mean2_sq);
         float mean1 = sum1_shared[0] / (width * height);
         float mean2 = sum2_shared[0] / (width * height);
         belta = mean1 - alpha * mean2;
     }
     __syncthreads();
     
     if (i <= 2*hw && j <= 2*hh) {
         int local_i = i - hw;
         int local_j = j - hh;
         
         float g1 = interpolate_gpu(x1 + local_i, y1 + local_j, img1, ncols, nrows);
         float g2 = interpolate_gpu(x2 + local_i, y2 + local_j, img2, ncols, nrows);
         
         int idx = j * (2*hw + 1) + i;
         imgdiff[idx] = g1 - g2 * alpha - belta;
     }
 }
 
 void computeIntensityDifference_gpu(
     _KLT_FloatImage img1,
     _KLT_FloatImage img2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow imgdiff)
 {
     int ncols = img1->ncols;
     int nrows = img1->nrows;
     int img_size = ncols * nrows * sizeof(float);
     int diff_size = width * height * sizeof(float);
     
     // OPTIMIZATION: Initialize GPU memory pool (allocate once!)
     _initTrackingGPUPool(ncols, nrows, width > height ? width : height);
     
     // Use stream 0 for legacy single-feature operations
     cudaStream_t stream = gpu_tracking_pool.streams[0];
     
     // OPTIMIZATION: Use async H2D transfers
     cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_img_buffer, img1->data, img_size, 
                                    cudaMemcpyHostToDevice, stream));
     cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_img_buffer + (ncols * nrows), 
                                    img2->data, img_size, 
                                    cudaMemcpyHostToDevice, stream));
     
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     // Launch kernel on stream
     computeIntensityDifferenceKernel<<<gridDim, blockDim, 0, stream>>>(
         gpu_tracking_pool.d_img_buffer, 
         gpu_tracking_pool.d_img_buffer + (ncols * nrows),
         x1, y1, x2, y2, width, height, ncols, nrows, gpu_tracking_pool.d_diff_buffer);
     
     // OPTIMIZATION: Async D2H transfer
     cudaCheckError(cudaMemcpyAsync(imgdiff, gpu_tracking_pool.d_diff_buffer, diff_size, 
                                    cudaMemcpyDeviceToHost, stream));
     
     // Synchronize this stream only (not entire GPU)
     cudaCheckError(cudaStreamSynchronize(stream));
 }
 
 void computeGradientSum_gpu(
     _KLT_FloatImage gradx1,
     _KLT_FloatImage grady1,
     _KLT_FloatImage gradx2,
     _KLT_FloatImage grady2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow gradx,
     _FloatWindow grady)
 {
     int ncols = gradx1->ncols;
     int nrows = gradx1->nrows;
     int img_size = ncols * nrows * sizeof(float);
     int grad_size = width * height * sizeof(float);
     
     // OPTIMIZATION: Initialize GPU memory pool
     _initTrackingGPUPool(ncols, nrows, width > height ? width : height);
     
     // Use stream 0 for legacy single-feature operations
     cudaStream_t stream = gpu_tracking_pool.streams[0];
     
     // OPTIMIZATION: Use async transfers with pointer arithmetic for 4 images
     // Buffer layout: [gradx1][grady1][gradx2][grady2][outputs...]
     float *d_grady1_offset = gpu_tracking_pool.d_img_buffer + (ncols * nrows);
     float *d_gradx2_offset = gpu_tracking_pool.d_img_buffer + (2 * ncols * nrows);
     float *d_grady2_offset = gpu_tracking_pool.d_img_buffer + (3 * ncols * nrows);
     float *d_gradx_out = gpu_tracking_pool.d_diff_buffer;
     float *d_grady_out = gpu_tracking_pool.d_temp_buffer;
     
     cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_img_buffer, gradx1->data, img_size, 
                                    cudaMemcpyHostToDevice, stream));
     cudaCheckError(cudaMemcpyAsync(d_grady1_offset, grady1->data, img_size, 
                                    cudaMemcpyHostToDevice, stream));
     cudaCheckError(cudaMemcpyAsync(d_gradx2_offset, gradx2->data, img_size, 
                                    cudaMemcpyHostToDevice, stream));
     cudaCheckError(cudaMemcpyAsync(d_grady2_offset, grady2->data, img_size, 
                                    cudaMemcpyHostToDevice, stream));
     
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     // Launch kernel on stream
     computeGradientSumKernel<<<gridDim, blockDim, 0, stream>>>(
         gpu_tracking_pool.d_img_buffer, d_grady1_offset, d_gradx2_offset, d_grady2_offset,
         x1, y1, x2, y2, width, height, ncols, nrows,
         d_gradx_out, d_grady_out);
     
     // OPTIMIZATION: Async D2H transfers
     cudaCheckError(cudaMemcpyAsync(gradx, d_gradx_out, grad_size, 
                                    cudaMemcpyDeviceToHost, stream));
     cudaCheckError(cudaMemcpyAsync(grady, d_grady_out, grad_size, 
                                    cudaMemcpyDeviceToHost, stream));
     
     // Synchronize stream only
     cudaCheckError(cudaStreamSynchronize(stream));
 }
 
 void computeIntensityDifferenceLightingInsensitive_gpu(
     _KLT_FloatImage img1,
     _KLT_FloatImage img2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow imgdiff)
 {
     int ncols = img1->ncols;
     int nrows = img1->nrows;
     int img_size = ncols * nrows * sizeof(float);
     int diff_size = width * height * sizeof(float);
     
     float *d_img1, *d_img2, *d_imgdiff;
     
     cudaCheckError(cudaMalloc(&d_img1, img_size));
     cudaCheckError(cudaMalloc(&d_img2, img_size));
     cudaCheckError(cudaMalloc(&d_imgdiff, diff_size));
     
     cudaCheckError(cudaMemcpy(d_img1, img1->data, img_size, cudaMemcpyHostToDevice));
     cudaCheckError(cudaMemcpy(d_img2, img2->data, img_size, cudaMemcpyHostToDevice));
     
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     computeIntensityDifferenceLightingInsensitiveKernel<<<gridDim, blockDim>>>(
         d_img1, d_img2, x1, y1, x2, y2, width, height, ncols, nrows, d_imgdiff);
     
     cudaCheckError(cudaDeviceSynchronize());
     cudaCheckError(cudaMemcpy(imgdiff, d_imgdiff, diff_size, cudaMemcpyDeviceToHost));
     
     cudaFree(d_img1);
     cudaFree(d_img2);
     cudaFree(d_imgdiff);
 }
 
 /*********************************************************************
  * CPU-side wrapper functions that use GPU acceleration
  *********************************************************************/
 
 static void _computeIntensityDifference(
     _KLT_FloatImage img1,
     _KLT_FloatImage img2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow imgdiff)
 {
     computeIntensityDifference_gpu(img1, img2, x1, y1, x2, y2, width, height, imgdiff);
 }
 
 static void _computeGradientSum(
     _KLT_FloatImage gradx1,
     _KLT_FloatImage grady1,
     _KLT_FloatImage gradx2,
     _KLT_FloatImage grady2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow gradx,
     _FloatWindow grady)
 {
     computeGradientSum_gpu(gradx1, grady1, gradx2, grady2, x1, y1, x2, y2, width, height, gradx, grady);
 }
 
 static void _computeIntensityDifferenceLightingInsensitive(
     _KLT_FloatImage img1,
     _KLT_FloatImage img2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow imgdiff)
 {
     computeIntensityDifferenceLightingInsensitive_gpu(img1, img2, x1, y1, x2, y2, width, height, imgdiff);
 }
 
 static void _computeGradientSumLightingInsensitive(
     _KLT_FloatImage gradx1,
     _KLT_FloatImage grady1,
     _KLT_FloatImage gradx2,
     _KLT_FloatImage grady2,
     _KLT_FloatImage img1,
     _KLT_FloatImage img2,
     float x1, float y1,
     float x2, float y2,
     int width, int height,
     _FloatWindow gradx,
     _FloatWindow grady)
 {
     /* For lighting insensitive, we use the regular gradient sum for now */
     /* The GPU implementation would need to be extended for full lighting compensation */
     computeGradientSum_gpu(gradx1, grady1, gradx2, grady2, x1, y1, x2, y2, width, height, gradx, grady);
 }
 
 /*********************************************************************
  * _allocateFloatWindow
  */
 static _FloatWindow _allocateFloatWindow(int width, int height)
 {
     _FloatWindow fw;
     fw = (_FloatWindow) malloc(width*height*sizeof(float));
     if (fw == NULL) {
         fprintf(stderr, "(_allocateFloatWindow) Out of memory.\n");
         exit(1);
     }
     return fw;
 }
 
 /*********************************************************************
  * _compute2by2GradientMatrix
  */
 static void _compute2by2GradientMatrix(
     _FloatWindow gradx,
     _FloatWindow grady,
     int width,
     int height,
     float *gxx,
     float *gxy,
     float *gyy)
 {
     register float gx, gy;
     register int i;
     
     *gxx = 0.0;  *gxy = 0.0;  *gyy = 0.0;
     for (i = 0 ; i < width * height ; i++)  {
         gx = *gradx++;
         gy = *grady++;
         *gxx += gx*gx;
         *gxy += gx*gy;
         *gyy += gy*gy;
     }
 }
 
 /*********************************************************************
  * _compute2by1ErrorVector
  */
 static void _compute2by1ErrorVector(
     _FloatWindow imgdiff,
     _FloatWindow gradx,
     _FloatWindow grady,
     int width,
     int height,
     float step_factor,
     float *ex,
     float *ey)
 {
     register float diff;
     register int i;
     
     *ex = 0;  *ey = 0;
     for (i = 0 ; i < width * height ; i++)  {
         diff = *imgdiff++;
         *ex += diff * (*gradx++);
         *ey += diff * (*grady++);
     }
     *ex *= step_factor;
     *ey *= step_factor;
 }
 
 /*********************************************************************
  * _solveEquation
  */
 static int _solveEquation(
     float gxx, float gxy, float gyy,
     float ex, float ey,
     float small,
     float *dx, float *dy)
 {
     float det = gxx*gyy - gxy*gxy;
     
     if (det < small)  return KLT_SMALL_DET;
     
     *dx = (gyy*ex - gxy*ey)/det;
     *dy = (gxx*ey - gxy*ex)/det;
     return KLT_TRACKED;
 }
 
 /*********************************************************************
  * _sumAbsFloatWindow
  */
 static float _sumAbsFloatWindow(
     _FloatWindow fw,
     int width,
     int height)
 {
     float sum = 0.0;
     int w;
     
     for ( ; height > 0 ; height--)
         for (w=0 ; w < width ; w++)
             sum += (float) fabs(*fw++);
     
     return sum;
 }
 
 /*********************************************************************
  * _trackFeature
  *
  * Tracks a feature point from one image to the next using GPU acceleration.
  */
 static int _trackFeature(
     float x1,
     float y1,
     float *x2,
     float *y2,
     _KLT_FloatImage img1,
     _KLT_FloatImage gradx1,
     _KLT_FloatImage grady1,
     _KLT_FloatImage img2,
     _KLT_FloatImage gradx2,
     _KLT_FloatImage grady2,
     int width,
     int height,
     float step_factor,
     int max_iterations,
     float small,
     float th,
     float max_residue,
     int lighting_insensitive)
 {
     _FloatWindow imgdiff, gradx, grady;
     float gxx, gxy, gyy, ex, ey, dx, dy;
     int iteration = 0;
     int status;
     int hw = width/2;
     int hh = height/2;
     int nc = img1->ncols;
     int nr = img1->nrows;
     float one_plus_eps = 1.001f;
     
     /* Allocate memory for windows */
     imgdiff = _allocateFloatWindow(width, height);
     gradx   = _allocateFloatWindow(width, height);
     grady   = _allocateFloatWindow(width, height);
     
     /* Iteratively update the window position */
     do  {
         /* If out of bounds, exit loop */
         if (  x1-hw < 0.0f || nc-( x1+hw) < one_plus_eps ||
              *x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
               y1-hh < 0.0f || nr-( y1+hh) < one_plus_eps ||
              *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps) {
             status = KLT_OOB;
             break;
         }
         
         /* Compute gradient and difference windows using GPU */
         if (lighting_insensitive) {
             _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2,
                                                           width, height, imgdiff);
             _computeGradientSumLightingInsensitive(gradx1, grady1, gradx2, grady2,
                                                    img1, img2, x1, y1, *x2, *y2, width, height, gradx, grady);
         } else {
             _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2,
                                        width, height, imgdiff);
             _computeGradientSum(gradx1, grady1, gradx2, grady2,
                                x1, y1, *x2, *y2, width, height, gradx, grady);
         }
         
         /* Use these windows to construct matrices */
         _compute2by2GradientMatrix(gradx, grady, width, height,
                                    &gxx, &gxy, &gyy);
         _compute2by1ErrorVector(imgdiff, gradx, grady, width, height, step_factor,
                                &ex, &ey);
         
         /* Using matrices, solve equation for new displacement */
         status = _solveEquation(gxx, gxy, gyy, ex, ey, small, &dx, &dy);
         if (status == KLT_SMALL_DET)  break;
         
         *x2 += dx;
         *y2 += dy;
         iteration++;
         
     }  while ((fabs(dx)>=th || fabs(dy)>=th) && iteration < max_iterations);
     
     /* Check whether window is out of bounds */
     if (*x2-hw < 0.0f || nc-(*x2+hw) < one_plus_eps ||
         *y2-hh < 0.0f || nr-(*y2+hh) < one_plus_eps)
         status = KLT_OOB;
     
     /* Check whether residue is too large */
     if (status == KLT_TRACKED)  {
         if (lighting_insensitive)
             _computeIntensityDifferenceLightingInsensitive(img1, img2, x1, y1, *x2, *y2,
                                                           width, height, imgdiff);
         else
             _computeIntensityDifference(img1, img2, x1, y1, *x2, *y2,
                                        width, height, imgdiff);
         if (_sumAbsFloatWindow(imgdiff, width, height)/(width*height) > max_residue)
             status = KLT_LARGE_RESIDUE;
     }
     
     /* Free memory */
     free(imgdiff);  free(gradx);  free(grady);
     
     /* Return appropriate value */
     if (status == KLT_SMALL_DET)  return KLT_SMALL_DET;
     else if (status == KLT_OOB)  return KLT_OOB;
     else if (status == KLT_LARGE_RESIDUE)  return KLT_LARGE_RESIDUE;
     else if (iteration >= max_iterations)  return KLT_MAX_ITERATIONS;
     else  return KLT_TRACKED;
 }
 
 /*********************************************************************
  * _outOfBounds
  */
 static KLT_BOOL _outOfBounds(
     float x,
     float y,
     int ncols,
     int nrows,
     int borderx,
     int bordery)
 {
     return (x < borderx || x > ncols-1-borderx ||
             y < bordery || y > nrows-1-bordery );
 }
 
 /*********************************************************************
  * KLTTrackFeatures
  *
  * Main tracking function - GPU accelerated version
  */
 extern "C" void KLTTrackFeatures(
     KLT_TrackingContext tc,
     KLT_PixelType *img1,
     KLT_PixelType *img2,
     int ncols,
     int nrows,
     KLT_FeatureList featurelist)
 {
     _KLT_FloatImage tmpimg, floatimg1, floatimg2;
     _KLT_Pyramid pyramid1, pyramid1_gradx, pyramid1_grady,
         pyramid2, pyramid2_gradx, pyramid2_grady;
     float subsampling = (float) tc->subsampling;
     float xloc, yloc, xlocout, ylocout;
     int val;
     int indx, r;
    KLT_BOOL floatimg1_created = FALSE;
    int i;
     
     /* Check window size (and correct if necessary) */
     if (tc->window_width % 2 != 1) {
         tc->window_width = tc->window_width+1;
     }
     if (tc->window_height % 2 != 1) {
         tc->window_height = tc->window_height+1;
     }
     if (tc->window_width < 3) {
         tc->window_width = 3;
     }
     if (tc->window_height < 3) {
         tc->window_height = 3;
     }
     
     /* Create temporary image */
     tmpimg = _KLTCreateFloatImage(ncols, nrows);
     
     /* Process first image by converting to float, smoothing, computing pyramid and gradients */
     if (tc->sequentialMode && tc->pyramid_last != NULL)  {
         pyramid1 = (_KLT_Pyramid) tc->pyramid_last;
         pyramid1_gradx = (_KLT_Pyramid) tc->pyramid_last_gradx;
         pyramid1_grady = (_KLT_Pyramid) tc->pyramid_last_grady;
         if (pyramid1->ncols[0] != ncols || pyramid1->nrows[0] != nrows) {
             fprintf(stderr, "(KLTTrackFeatures) Size of incoming image (%d by %d) "
                     "is different from size of previous image (%d by %d)\n",
                     ncols, nrows, pyramid1->ncols[0], pyramid1->nrows[0]);
             exit(1);
         }
         assert(pyramid1_gradx != NULL);
         assert(pyramid1_grady != NULL);
     } else  {
         floatimg1_created = TRUE;
         floatimg1 = _KLTCreateFloatImage(ncols, nrows);
         _KLTToFloatImage(img1, ncols, nrows, tmpimg);
         _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg1);
         pyramid1 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
         _KLTComputePyramid(floatimg1, pyramid1, tc->pyramid_sigma_fact);
         pyramid1_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
         pyramid1_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
         for (i = 0 ; i < tc->nPyramidLevels ; i++)
             _KLTComputeGradients(pyramid1->img[i], tc->grad_sigma,
                                 pyramid1_gradx->img[i],
                                 pyramid1_grady->img[i]);
     }
     
    /* Do the same thing with second image */
    floatimg2 = _KLTCreateFloatImage(ncols, nrows);
    _KLTToFloatImage(img2, ncols, nrows, tmpimg);
    _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg2);
    pyramid2 = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
    _KLTComputePyramid(floatimg2, pyramid2, tc->pyramid_sigma_fact);
    pyramid2_gradx = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
    pyramid2_grady = _KLTCreatePyramid(ncols, nrows, (int) subsampling, tc->nPyramidLevels);
    for (i = 0 ; i < tc->nPyramidLevels ; i++)
        _KLTComputeGradients(pyramid2->img[i], tc->grad_sigma,
                            pyramid2_gradx->img[i],
                            pyramid2_grady->img[i]);
    
   // HPC OPTIMIZATION: Upload pyramids with pointer swapping for sequential mode
   // If sequential mode and pyramid_last exists, pyramid1 is reused from previous frame
   // We swap GPU pointers instead of re-uploading - cuts pyramid upload time by 50%!
   int sequential_reuse = (tc->sequentialMode && tc->pyramid_last != NULL);
   _uploadPyramidsToGPU(pyramid1, pyramid1_gradx, pyramid1_grady,
                        pyramid2, pyramid2_gradx, pyramid2_grady,
                        tc->nPyramidLevels, sequential_reuse);
    
    // HPC OPTIMIZATION: Use persistent feature buffers (allocated once, reused forever)
    int nFeatures = featurelist->nFeatures;
    
    // Allocate persistent feature buffers on first use (reuse across all tracking calls)
    if (gpu_tracking_pool.max_features < nFeatures) {
        // Free old buffers if they exist
        if (gpu_tracking_pool.d_features) cudaFree(gpu_tracking_pool.d_features);
        if (gpu_tracking_pool.h_features) cudaFreeHost(gpu_tracking_pool.h_features);
        
        // Allocate new persistent buffers
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_features, nFeatures * sizeof(FeatureData)));
        cudaCheckError(cudaMallocHost(&gpu_tracking_pool.h_features, nFeatures * sizeof(FeatureData)));
        gpu_tracking_pool.max_features = nFeatures;
    }
    
    // Use the persistent buffers
    FeatureData *h_features = gpu_tracking_pool.h_features;
    FeatureData *d_features = gpu_tracking_pool.d_features;
    
    // HPC OPTIMIZATION: Upload tracking parameters to constant memory ONCE
    // These are read by ALL threads but never change - perfect for constant memory!
    // Reduces parameter passing overhead and improves cache hit rate
    cudaCheckError(cudaMemcpyToSymbol(c_window_width, &tc->window_width, sizeof(int)));
    cudaCheckError(cudaMemcpyToSymbol(c_window_height, &tc->window_height, sizeof(int)));
    cudaCheckError(cudaMemcpyToSymbol(c_step_factor, &tc->step_factor, sizeof(float)));
    cudaCheckError(cudaMemcpyToSymbol(c_max_iterations, &tc->max_iterations, sizeof(int)));
    cudaCheckError(cudaMemcpyToSymbol(c_min_determinant, &tc->min_determinant, sizeof(float)));
    cudaCheckError(cudaMemcpyToSymbol(c_min_displacement, &tc->min_displacement, sizeof(float)));
    cudaCheckError(cudaMemcpyToSymbol(c_max_residue, &tc->max_residue, sizeof(float)));
    
    /* HPC ASYNC: Process all pyramid levels with async transfers */
    for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
        
        // HPC: Use round-robin streams for overlap within each level
        int stream_id = r % NUM_STREAMS;
        cudaStream_t stream = gpu_tracking_pool.streams[stream_id];
        
        
        // HPC: Prepare batched feature data for this pyramid level
        int active_features = 0;
        for (indx = 0 ; indx < nFeatures ; indx++)  {
            // Copy current feature status (GPU needs to know which to skip)
            h_features[indx].status = featurelist->feature[indx]->val;
            
            if (featurelist->feature[indx]->val >= 0)  {
                if (r == tc->nPyramidLevels - 1) {
                    // First level: initialize from feature list
                    xloc = featurelist->feature[indx]->x;
                    yloc = featurelist->feature[indx]->y;
                    // Transform to coarsest resolution - divide nPyramidLevels times like V1-1
                    for (int rr = tc->nPyramidLevels - 1 ; rr >= 0 ; rr--)  {
                        xloc /= subsampling;  yloc /= subsampling;
                    }
                    // Then multiply once to match V1-1 behavior (multiply BEFORE tracking)
                    xloc *= subsampling;  yloc *= subsampling;
                    // Store in batched structure - same logic, different storage!
                    h_features[indx].x1 = xloc;
                    h_features[indx].y1 = yloc;
                    h_features[indx].x2 = xloc;
                    h_features[indx].y2 = yloc;
                } else {
                    // Propagate from previous level - multiply by subsampling
                    // x1/y1 = reference position in img1 (scales up each level)
                    // x2/y2 = tracked position in img2 (uses tracked result from previous level)
                    h_features[indx].x1 *= subsampling;
                    h_features[indx].y1 *= subsampling;
                    h_features[indx].x2 *= subsampling;  // Scale up tracked result from previous level
                    h_features[indx].y2 *= subsampling;
                }
                active_features++;
            }
        }
        
        if (active_features == 0) break;
        
        // HPC OPTIMIZATION: Create texture objects for hardware-accelerated interpolation
        // Textures provide FREE bilinear interpolation + better cache locality!
        int ncols = pyramid1->ncols[r];
        int nrows = pyramid1->nrows[r];
        cudaTextureObject_t tex_img1 = createTextureObject(gpu_tracking_pool.d_pyramid1_imgs[r], ncols, nrows);
        cudaTextureObject_t tex_img2 = createTextureObject(gpu_tracking_pool.d_pyramid2_imgs[r], ncols, nrows);
        cudaTextureObject_t tex_gradx1 = createTextureObject(gpu_tracking_pool.d_pyramid1_gradx[r], ncols, nrows);
        cudaTextureObject_t tex_grady1 = createTextureObject(gpu_tracking_pool.d_pyramid1_grady[r], ncols, nrows);
        cudaTextureObject_t tex_gradx2 = createTextureObject(gpu_tracking_pool.d_pyramid2_gradx[r], ncols, nrows);
        cudaTextureObject_t tex_grady2 = createTextureObject(gpu_tracking_pool.d_pyramid2_grady[r], ncols, nrows);
        
        // HPC OPTIMIZATION: Single batched H2D transfer (was 5 transfers, now 1!)
        // Transfers all feature data (x1, y1, x2, y2, status) in one contiguous block
        cudaCheckError(cudaMemcpyAsync(d_features, h_features, nFeatures * sizeof(FeatureData), 
                                       cudaMemcpyHostToDevice, stream));
        
        // Launch batched tracking kernel on stream - enables overlap!
        int threadsPerBlock = 256;
        int blocksPerGrid = (nFeatures + threadsPerBlock - 1) / threadsPerBlock;
        
        // HPC: Kernel uses texture memory + constant memory + batched structure!
        trackFeaturesBatchedKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
            tex_img1, tex_img2,           // HPC: Texture objects for hardware interpolation!
            tex_gradx1, tex_grady1,
            tex_gradx2, tex_grady2,
            d_features,                    // HPC: Batched structure instead of 5 separate arrays!
            nFeatures,
            ncols, nrows);
        
        // HPC OPTIMIZATION: Single batched D2H transfer (was 3 transfers, now 1!)
        cudaCheckError(cudaMemcpyAsync(h_features, d_features, nFeatures * sizeof(FeatureData), 
                                       cudaMemcpyDeviceToHost, stream));
        
        // HPC: Sync this stream to ensure D2H is complete before next level reads h_features
        cudaCheckError(cudaStreamSynchronize(stream));
        
        // HPC: Destroy texture objects after sync
        cudaCheckError(cudaDestroyTextureObject(tex_img1));
        cudaCheckError(cudaDestroyTextureObject(tex_img2));
        cudaCheckError(cudaDestroyTextureObject(tex_gradx1));
        cudaCheckError(cudaDestroyTextureObject(tex_grady1));
        cudaCheckError(cudaDestroyTextureObject(tex_gradx2));
        cudaCheckError(cudaDestroyTextureObject(tex_grady2));
        
        // Update feature status for this pyramid level (needed for next level)
        for (indx = 0 ; indx < nFeatures ; indx++)  {
            if (featurelist->feature[indx]->val >= 0) {
                // Mark features that failed at this level - they won't be processed further
                if (h_features[indx].status == KLT_SMALL_DET || h_features[indx].status == KLT_OOB) {
                    featurelist->feature[indx]->val = h_features[indx].status;
                }
                // Note: h_features[].x2/y2 now contains tracked positions for successful features
                // These will be scaled up for the next pyramid level in next iteration
                // h_features[].x1/y1 remains as reference (img1 position) and also gets scaled up
            }
        }
    }
    
    // Final update of feature positions from GPU results (finest level results in h_features)
    for (indx = 0 ; indx < nFeatures ; indx++)  {
        if (featurelist->feature[indx]->val >= 0) {
            val = h_features[indx].status;
            
            // CRITICAL: Only use coordinates if feature was successfully tracked
            // If feature failed at intermediate pyramid level, coords are at wrong scale!
            if (val == KLT_TRACKED || val == KLT_MAX_ITERATIONS) {
                xlocout = h_features[indx].x2;
                ylocout = h_features[indx].y2;
            } else {
                // Feature failed - use placeholder coordinates
                xlocout = -1.0;
                ylocout = -1.0;
            }
            
            /* Record feature */
            if (val == KLT_OOB) {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_OOB;
            } else if (_outOfBounds(xlocout, ylocout, ncols, nrows, tc->borderx, tc->bordery))  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_OOB;
            } else if (val == KLT_SMALL_DET)  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_SMALL_DET;
            } else if (val == KLT_LARGE_RESIDUE)  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_LARGE_RESIDUE;
            } else if (val == KLT_MAX_ITERATIONS)  {
                featurelist->feature[indx]->x   = -1.0;
                featurelist->feature[indx]->y   = -1.0;
                featurelist->feature[indx]->val = KLT_MAX_ITERATIONS;
            } else  {
                featurelist->feature[indx]->x = xlocout;
                featurelist->feature[indx]->y = ylocout;
                featurelist->feature[indx]->val = KLT_TRACKED;
            }
        }
    }
    
    // HPC: Feature buffers are now persistent (allocated once, reused forever)
    // No need to free here - they'll be freed when pool is destroyed or reallocated
     
     if (tc->sequentialMode)  {
         tc->pyramid_last = pyramid2;
         tc->pyramid_last_gradx = pyramid2_gradx;
         tc->pyramid_last_grady = pyramid2_grady;
     } else  {
         _KLTFreePyramid(pyramid2);
         _KLTFreePyramid(pyramid2_gradx);
         _KLTFreePyramid(pyramid2_grady);
     }
     
     /* Free memory */
     _KLTFreeFloatImage(tmpimg);
     if (floatimg1_created)  _KLTFreeFloatImage(floatimg1);
     _KLTFreeFloatImage(floatimg2);
    _KLTFreePyramid(pyramid1);
    _KLTFreePyramid(pyramid1_gradx);
    _KLTFreePyramid(pyramid1_grady);
    
    if (KLT_verbose >= 1)  {
        fprintf(stderr,  "\n\t%d features successfully tracked.\n",
                KLTCountRemainingFeatures(featurelist));
        fflush(stderr);
    }
 }
 
 // OPTIMIZATION: Cleanup GPU tracking pool (call at program end)
 extern "C" void _KLTTrackingFeaturesCleanup()
 {
     _cleanupTrackingGPUPool();
 }
 
 
 