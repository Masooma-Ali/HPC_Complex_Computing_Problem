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

// CUDA error checking macro and helper function
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// OPTIMIZATION: GPU Memory Pool for persistent allocation
typedef struct {
    float *d_img_buffer;           // Reusable for input images
    float *d_diff_buffer;          // Reusable for difference windows  
    float *d_temp_buffer;          // Reusable for temporary data
    cudaStream_t stream;           // For async operations
    int allocated_img_size;        // Maximum image buffer size
    int allocated_diff_size;       // Maximum difference buffer size
    int initialized;               // Flag to check if initialized
} GPUTrackingMemoryPool;

static GPUTrackingMemoryPool gpu_tracking_pool = {NULL, NULL, NULL, NULL, 0, 0, 0};

// OPTIMIZATION: Initialize GPU memory pool (allocate once, reuse forever)
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
        if (gpu_tracking_pool.stream) cudaStreamDestroy(gpu_tracking_pool.stream);
        
        // Allocate persistent GPU memory (allocate once!)
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_img_buffer, img_size * 4)); // For 4 gradient images
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_diff_buffer, diff_size));
        cudaCheckError(cudaMalloc(&gpu_tracking_pool.d_temp_buffer, diff_size));
        
        // Create stream for async operations
        cudaCheckError(cudaStreamCreate(&gpu_tracking_pool.stream));
        
        gpu_tracking_pool.allocated_img_size = img_size;
        gpu_tracking_pool.allocated_diff_size = diff_size;
        gpu_tracking_pool.initialized = 1;
    }
}

// OPTIMIZATION: Cleanup GPU pool
void _cleanupTrackingGPUPool()
{
    if (gpu_tracking_pool.initialized) {
        if (gpu_tracking_pool.d_img_buffer) cudaFree(gpu_tracking_pool.d_img_buffer);
        if (gpu_tracking_pool.d_diff_buffer) cudaFree(gpu_tracking_pool.d_diff_buffer);
        if (gpu_tracking_pool.d_temp_buffer) cudaFree(gpu_tracking_pool.d_temp_buffer);
        if (gpu_tracking_pool.stream) cudaStreamDestroy(gpu_tracking_pool.stream);
        
        gpu_tracking_pool.d_img_buffer = NULL;
        gpu_tracking_pool.d_diff_buffer = NULL;
        gpu_tracking_pool.d_temp_buffer = NULL;
        gpu_tracking_pool.initialized = 0;
    }
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
     
     // OPTIMIZATION: Use async H2D transfers
     cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_img_buffer, img1->data, img_size, 
                                    cudaMemcpyHostToDevice, gpu_tracking_pool.stream));
     cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_img_buffer + (ncols * nrows), 
                                    img2->data, img_size, 
                                    cudaMemcpyHostToDevice, gpu_tracking_pool.stream));
     
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     // Launch kernel on stream
     computeIntensityDifferenceKernel<<<gridDim, blockDim, 0, gpu_tracking_pool.stream>>>(
         gpu_tracking_pool.d_img_buffer, 
         gpu_tracking_pool.d_img_buffer + (ncols * nrows),
         x1, y1, x2, y2, width, height, ncols, nrows, gpu_tracking_pool.d_diff_buffer);
     
     // OPTIMIZATION: Async D2H transfer
     cudaCheckError(cudaMemcpyAsync(imgdiff, gpu_tracking_pool.d_diff_buffer, diff_size, 
                                    cudaMemcpyDeviceToHost, gpu_tracking_pool.stream));
     
     // Synchronize this stream only (not entire GPU)
     cudaCheckError(cudaStreamSynchronize(gpu_tracking_pool.stream));
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
     
     // OPTIMIZATION: Use async transfers with pointer arithmetic for 4 images
     // Buffer layout: [gradx1][grady1][gradx2][grady2][outputs...]
     float *d_grady1_offset = gpu_tracking_pool.d_img_buffer + (ncols * nrows);
     float *d_gradx2_offset = gpu_tracking_pool.d_img_buffer + (2 * ncols * nrows);
     float *d_grady2_offset = gpu_tracking_pool.d_img_buffer + (3 * ncols * nrows);
     float *d_gradx_out = gpu_tracking_pool.d_diff_buffer;
     float *d_grady_out = gpu_tracking_pool.d_temp_buffer;
     
     cudaCheckError(cudaMemcpyAsync(gpu_tracking_pool.d_img_buffer, gradx1->data, img_size, 
                                    cudaMemcpyHostToDevice, gpu_tracking_pool.stream));
     cudaCheckError(cudaMemcpyAsync(d_grady1_offset, grady1->data, img_size, 
                                    cudaMemcpyHostToDevice, gpu_tracking_pool.stream));
     cudaCheckError(cudaMemcpyAsync(d_gradx2_offset, gradx2->data, img_size, 
                                    cudaMemcpyHostToDevice, gpu_tracking_pool.stream));
     cudaCheckError(cudaMemcpyAsync(d_grady2_offset, grady2->data, img_size, 
                                    cudaMemcpyHostToDevice, gpu_tracking_pool.stream));
     
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     // Launch kernel on stream
     computeGradientSumKernel<<<gridDim, blockDim, 0, gpu_tracking_pool.stream>>>(
         gpu_tracking_pool.d_img_buffer, d_grady1_offset, d_gradx2_offset, d_grady2_offset,
         x1, y1, x2, y2, width, height, ncols, nrows,
         d_gradx_out, d_grady_out);
     
     // OPTIMIZATION: Async D2H transfers
     cudaCheckError(cudaMemcpyAsync(gradx, d_gradx_out, grad_size, 
                                    cudaMemcpyDeviceToHost, gpu_tracking_pool.stream));
     cudaCheckError(cudaMemcpyAsync(grady, d_grady_out, grad_size, 
                                    cudaMemcpyDeviceToHost, gpu_tracking_pool.stream));
     
     // Synchronize stream only
     cudaCheckError(cudaStreamSynchronize(gpu_tracking_pool.stream));
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
     
     if (KLT_verbose >= 1)  {
         fprintf(stderr,  "(KLT-GPU) Tracking %d features in a %d by %d image...  ",
                 KLTCountRemainingFeatures(featurelist), ncols, nrows);
         fflush(stderr);
     }
     
     /* Check window size (and correct if necessary) */
     if (tc->window_width % 2 != 1) {
         tc->window_width = tc->window_width+1;
         fprintf(stderr, "Tracking context's window width must be odd. Changing to %d.\n", 
                 tc->window_width);
     }
     if (tc->window_height % 2 != 1) {
         tc->window_height = tc->window_height+1;
         fprintf(stderr, "Tracking context's window height must be odd. Changing to %d.\n",
                 tc->window_height);
     }
     if (tc->window_width < 3) {
         tc->window_width = 3;
         fprintf(stderr, "Tracking context's window width must be at least three. Changing to %d.\n",
                 tc->window_width);
     }
     if (tc->window_height < 3) {
         tc->window_height = 3;
         fprintf(stderr, "Tracking context's window height must be at least three. Changing to %d.\n",
                 tc->window_height);
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
     
     /* For each feature, do ... */
     for (indx = 0 ; indx < featurelist->nFeatures ; indx++)  {
         
         /* Only track features that are not lost */
         if (featurelist->feature[indx]->val >= 0)  {
             
             xloc = featurelist->feature[indx]->x;
             yloc = featurelist->feature[indx]->y;
             
             /* Transform location to coarsest resolution */
             for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
                 xloc /= subsampling;  yloc /= subsampling;
             }
             xlocout = xloc;  ylocout = yloc;
             
             /* Beginning with coarsest resolution, do ... */
             for (r = tc->nPyramidLevels - 1 ; r >= 0 ; r--)  {
                 
                 /* Track feature at current resolution */
                 xloc *= subsampling;  yloc *= subsampling;
                 xlocout *= subsampling;  ylocout *= subsampling;
                 
                 val = _trackFeature(xloc, yloc,
                                    &xlocout, &ylocout,
                                    pyramid1->img[r],
                                    pyramid1_gradx->img[r], pyramid1_grady->img[r],
                                    pyramid2->img[r],
                                    pyramid2_gradx->img[r], pyramid2_grady->img[r],
                                    tc->window_width, tc->window_height,
                                    tc->step_factor,
                                    tc->max_iterations,
                                    tc->min_determinant,
                                    tc->min_displacement,
                                    tc->max_residue,
                                    tc->lighting_insensitive);
                 
                 if (val==KLT_SMALL_DET || val==KLT_OOB)
                     break;
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
         fprintf(stderr,  "\n\t%d features successfully tracked (GPU).\n",
                 KLTCountRemainingFeatures(featurelist));
         fflush(stderr);
     }
 }
 
 // OPTIMIZATION: Cleanup GPU tracking pool (call at program end)
 extern "C" void _KLTTrackingFeaturesCleanup()
 {
     _cleanupTrackingGPUPool();
 }
 
 
 