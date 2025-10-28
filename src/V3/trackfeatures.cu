/*********************************************************************
 * trackFeatures.cu - CUDA GPU-accelerated version
 * Complete implementation with CPU wrapper functions
 *********************************************************************/

#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
 
 /* Our includes */
 extern "C" {
 #include "base.h"
 #include "convolve.h"
 #include "klt.h"
 #include "klt_util.h"
 #include "pyramid.h"
 }
 
 #include "gpu_memory_pool.h"
 
 extern int KLT_verbose;
 
 #define BLOCK_SIZE 16
 #define MAX_FEATURES 1000
 
 typedef float *_FloatWindow;
 
 #define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
 inline void cudaAssert(cudaError_t code, const char *file, int line)
 {
     if (code != cudaSuccess) {
         fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
         exit(code);
     }
 }
 
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
    
    return ((1-ax) * (1-ay) * ptr[0] +
            ax * (1-ay) * ptr[1] +
            (1-ax) * ay * ptr[ncols] +
            ax * ay * ptr[ncols+1]);
}

/*********************************************************************
 * BATCHED KERNEL: Process all features at once
 * Each thread block processes one feature
 * Processes multiple features in parallel
 *********************************************************************/
__global__ void batchedComputeWindowsKernel(
    float *img1,
    float *img2,
    float *gradx1,
    float *grady1,
    float *gradx2,
    float *grady2,
    float *x1_in,
    float *y1_in,
    float *x2_in,
    float *y2_in,
    int n_features,
    int window_width,
    int window_height,
    int ncols,
    int nrows,
    float *imgdiff_out,    // [n_features * window_size]
    float *gradx_out,      // [n_features * window_size]
    float *grady_out)      // [n_features * window_size]
{
    // Each block processes one feature
    int feature_idx = blockIdx.x;
    if (feature_idx >= n_features) return;
    
    float x1 = x1_in[feature_idx];
    float y1 = y1_in[feature_idx];
    float x2 = x2_in[feature_idx];
    float y2 = y2_in[feature_idx];
    
    int hw = window_width / 2;
    int hh = window_height / 2;
    int window_size = window_width * window_height;
    int offset = feature_idx * window_size;
    
    // Each thread processes one pixel in the window
    int i = threadIdx.x;
    int j = threadIdx.y;
    
    if (i <= 2*hw && j <= 2*hh) {
        int local_i = i - hw;
        int local_j = j - hh;
        
        // Compute interpolated values
        float g1 = interpolate_gpu(x1 + local_i, y1 + local_j, img1, ncols, nrows);
        float g2 = interpolate_gpu(x2 + local_i, y2 + local_j, img2, ncols, nrows);
        
        float gx1 = interpolate_gpu(x1 + local_i, y1 + local_j, gradx1, ncols, nrows);
        float gx2 = interpolate_gpu(x2 + local_i, y2 + local_j, gradx2, ncols, nrows);
        float gy1 = interpolate_gpu(x1 + local_i, y1 + local_j, grady1, ncols, nrows);
        float gy2 = interpolate_gpu(x2 + local_i, y2 + local_j, grady2, ncols, nrows);
        
        int idx = j * (2*hw + 1) + i;
        imgdiff_out[offset + idx] = g1 - g2;
        gradx_out[offset + idx] = gx1 + gx2;
        grady_out[offset + idx] = gy1 + gy2;
    }
}

/*********************************************************************
 * BATCHED KERNEL: Compute gradient matrices for all features
 * Each thread processes one feature's window
 *********************************************************************/
__global__ void batchedComputeGradientMatricesKernel(
    float *gradx,
    float *grady,
    int n_features,
    int window_width,
    int window_height,
    float *gxx_out,
    float *gxy_out,
    float *gyy_out)
{
    int feature_idx = blockIdx.x;
    if (feature_idx >= n_features) return;
    
    int window_size = window_width * window_height;
    int offset = feature_idx * window_size;
    
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    // Each thread sums up all pixels in the window
    for (int i = threadIdx.x; i < window_size; i += blockDim.x) {
        float gx = gradx[offset + i];
        float gy = grady[offset + i];
        gxx += gx * gx;
        gxy += gx * gy;
        gyy += gy * gy;
    }
    
    // Reduction in shared memory
    __shared__ float s_gxx[256];
    __shared__ float s_gxy[256];
    __shared__ float s_gyy[256];
    
    int tid = threadIdx.x;
    s_gxx[tid] = gxx;
    s_gxy[tid] = gxy;
    s_gyy[tid] = gyy;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_gxx[tid] += s_gxx[tid + s];
            s_gxy[tid] += s_gxy[tid + s];
            s_gyy[tid] += s_gyy[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        gxx_out[feature_idx] = s_gxx[0];
        gxy_out[feature_idx] = s_gxy[0];
        gyy_out[feature_idx] = s_gyy[0];
    }
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
    int diff_size = width * height * sizeof(float);
    
    // Use pre-allocated buffers from memory pool
    float *d_img1 = GPU_GetImageBuffer(0);  // img1
    float *d_img2 = GPU_GetImageBuffer(1);  // img2  
    float *d_imgdiff = GPU_GetWindowBuffer(0, width, height);
    
    // Check if images are already on GPU (from pyramid upload)
    // If img1->data points to pinned memory, it might already be on GPU
    // For now, we'll use async copy to pinned memory
    int img_size = ncols * nrows * sizeof(float);
    
    // Use pinned memory if available for faster transfer
    float *h_pinned1 = GPU_GetPinnedHostBuffer(0);
    float *h_pinned2 = GPU_GetPinnedHostBuffer(1);
    
    if (h_pinned1 && h_pinned2) {
        // Copy to pinned memory first (fast)
        memcpy(h_pinned1, img1->data, img_size);
        memcpy(h_pinned2, img2->data, img_size);
        // Then async copy to GPU (faster than regular memory)
        cudaCheckError(cudaMemcpyAsync(d_img1, h_pinned1, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_img2, h_pinned2, img_size, cudaMemcpyHostToDevice));
    } else {
        // Fallback to regular async copy
        cudaCheckError(cudaMemcpyAsync(d_img1, img1->data, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_img2, img2->data, img_size, cudaMemcpyHostToDevice));
    }
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeIntensityDifferenceKernel<<<gridDim, blockDim>>>(
        d_img1, d_img2, x1, y1, x2, y2, width, height, ncols, nrows, d_imgdiff);
    
    // Copy only the small window result back
    cudaCheckError(cudaMemcpyAsync(imgdiff, d_imgdiff, diff_size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaDeviceSynchronize());
    
    // NO cudaFree - buffers are reused!
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
    
    // Use pre-allocated buffers from memory pool
    float *d_gradx1 = GPU_GetImageBuffer(2);  // gradx1
    float *d_grady1 = GPU_GetImageBuffer(3);  // grady1
    float *d_gradx2 = GPU_GetImageBuffer(4);  // gradx2
    float *d_grady2 = GPU_GetImageBuffer(5);  // grady2
    float *d_gradx_out = GPU_GetWindowBuffer(1, width, height);
    float *d_grady_out = GPU_GetWindowBuffer(2, width, height);
    
    // Use pinned memory for faster transfers
    float *h_pinned_gx1 = GPU_GetPinnedHostBuffer(2);
    float *h_pinned_gy1 = GPU_GetPinnedHostBuffer(3);
    float *h_pinned_gx2 = GPU_GetPinnedHostBuffer(4);
    float *h_pinned_gy2 = GPU_GetPinnedHostBuffer(5);
    
    if (h_pinned_gx1 && h_pinned_gy1 && h_pinned_gx2 && h_pinned_gy2) {
        memcpy(h_pinned_gx1, gradx1->data, img_size);
        memcpy(h_pinned_gy1, grady1->data, img_size);
        memcpy(h_pinned_gx2, gradx2->data, img_size);
        memcpy(h_pinned_gy2, grady2->data, img_size);
        cudaCheckError(cudaMemcpyAsync(d_gradx1, h_pinned_gx1, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_grady1, h_pinned_gy1, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_gradx2, h_pinned_gx2, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_grady2, h_pinned_gy2, img_size, cudaMemcpyHostToDevice));
    } else {
        cudaCheckError(cudaMemcpyAsync(d_gradx1, gradx1->data, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_grady1, grady1->data, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_gradx2, gradx2->data, img_size, cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpyAsync(d_grady2, grady2->data, img_size, cudaMemcpyHostToDevice));
    }
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeGradientSumKernel<<<gridDim, blockDim>>>(
        d_gradx1, d_grady1, d_gradx2, d_grady2,
        x1, y1, x2, y2, width, height, ncols, nrows,
        d_gradx_out, d_grady_out);
    
    // Copy only small window results back
    cudaCheckError(cudaMemcpyAsync(gradx, d_gradx_out, grad_size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpyAsync(grady, d_grady_out, grad_size, cudaMemcpyDeviceToHost));
    cudaCheckError(cudaDeviceSynchronize());
    
    // NO cudaFree - buffers are reused!
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
     
     // Use pre-allocated buffers from memory pool
     float *d_img1 = GPU_GetImageBuffer(0);
     float *d_img2 = GPU_GetImageBuffer(1);
     float *d_imgdiff = GPU_GetWindowBuffer(0, width, height);
     
     cudaCheckError(cudaMemcpyAsync(d_img1, img1->data, img_size, cudaMemcpyHostToDevice));
     cudaCheckError(cudaMemcpyAsync(d_img2, img2->data, img_size, cudaMemcpyHostToDevice));
     
     dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
     dim3 gridDim((width + BLOCK_SIZE - 1) / BLOCK_SIZE,
                  (height + BLOCK_SIZE - 1) / BLOCK_SIZE);
     
     computeIntensityDifferenceLightingInsensitiveKernel<<<gridDim, blockDim>>>(
         d_img1, d_img2, x1, y1, x2, y2, width, height, ncols, nrows, d_imgdiff);
     
     cudaCheckError(cudaMemcpyAsync(imgdiff, d_imgdiff, diff_size, cudaMemcpyDeviceToHost));
     cudaCheckError(cudaDeviceSynchronize());
     
     // NO cudaFree - buffers are reused!
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
    
    /* Upload pyramid levels to GPU ONCE - reuse for all features */
    for (i = 0 ; i < tc->nPyramidLevels ; i++) {
        // Upload pyramid1 level (if not already on GPU)
        if (!floatimg1_created || i == 0) {  // Only upload if newly created or first frame
            GPU_UploadPyramidLevel(0, i, 
                pyramid1->img[i]->data,
                pyramid1_gradx->img[i]->data,
                pyramid1_grady->img[i]->data,
                pyramid1->ncols[i], pyramid1->nrows[i]);
            GPU_SetPyramidDims(i, pyramid1->ncols[i], pyramid1->nrows[i]);
        }
        // Always upload pyramid2 (new frame)
        GPU_UploadPyramidLevel(1, i,
            pyramid2->img[i]->data,
            pyramid2_gradx->img[i]->data,
            pyramid2_grady->img[i]->data,
            pyramid2->ncols[i], pyramid2->nrows[i]);
    }
    // Synchronize to ensure uploads complete
    GPU_Sync();
    
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
 
 
 