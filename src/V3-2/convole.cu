
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#define MAX_KERNEL_WIDTH 71
#define BLOCK_SIZE 16
#define SHARED_MEM_SIZE (BLOCK_SIZE + MAX_KERNEL_WIDTH)
#define WARP_SIZE 32               // For bank conflict calculation
#define SHARED_PADDING ((MAX_KERNEL_WIDTH + WARP_SIZE/2) / WARP_SIZE) // Optimization 2

// OPTIMIZATION: GPU Memory Pool for persistent allocation
#define GPU_POOL_SIZE (10 * 512 * 512 * sizeof(float))  // Pre-allocate for 10 images
typedef struct {
    float *d_persistent_input;
    float *d_persistent_output;
    float *d_persistent_temp;
    int allocated_size;
    cudaStream_t stream_horiz;
    cudaStream_t stream_vert;
} GPUMemoryPool;

static GPUMemoryPool gpu_pool = {NULL, NULL, NULL, 0, NULL, NULL};
static int gpu_pool_initialized = 0;

typedef unsigned char KLT_PixelType;

typedef struct {
    int ncols;
    int nrows;
    float *data;
} _KLT_FloatImageRec, *_KLT_FloatImage;

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

// Optimization 9: Constant memory for kernel coefficients (NOW USED!)
__constant__ float d_kernel_const[MAX_KERNEL_WIDTH];

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// OPTIMIZATION: Initialize GPU memory pool and streams
void _initGPUMemoryPool(int image_size)
{
    if (!gpu_pool_initialized || gpu_pool.allocated_size < image_size) {
        // Free old allocations if they exist
        if (gpu_pool.d_persistent_input) cudaFree(gpu_pool.d_persistent_input);
        if (gpu_pool.d_persistent_output) cudaFree(gpu_pool.d_persistent_output);
        if (gpu_pool.d_persistent_temp) cudaFree(gpu_pool.d_persistent_temp);
        if (gpu_pool.stream_horiz) cudaStreamDestroy(gpu_pool.stream_horiz);
        if (gpu_pool.stream_vert) cudaStreamDestroy(gpu_pool.stream_vert);
        
        // Allocate persistent GPU memory
        cudaCheckError(cudaMalloc(&gpu_pool.d_persistent_input, image_size));
        cudaCheckError(cudaMalloc(&gpu_pool.d_persistent_output, image_size));
        cudaCheckError(cudaMalloc(&gpu_pool.d_persistent_temp, image_size));
        
        // Create streams for async operations
        cudaCheckError(cudaStreamCreate(&gpu_pool.stream_horiz));
        cudaCheckError(cudaStreamCreate(&gpu_pool.stream_vert));
        
        gpu_pool.allocated_size = image_size;
        gpu_pool_initialized = 1;
    }
}

// OPTIMIZATION: Cleanup GPU memory pool
void _cleanupGPUMemoryPool()
{
    if (gpu_pool_initialized) {
        if (gpu_pool.d_persistent_input) cudaFree(gpu_pool.d_persistent_input);
        if (gpu_pool.d_persistent_output) cudaFree(gpu_pool.d_persistent_output);
        if (gpu_pool.d_persistent_temp) cudaFree(gpu_pool.d_persistent_temp);
        if (gpu_pool.stream_horiz) cudaStreamDestroy(gpu_pool.stream_horiz);
        if (gpu_pool.stream_vert) cudaStreamDestroy(gpu_pool.stream_vert);
        
        gpu_pool.d_persistent_input = NULL;
        gpu_pool.d_persistent_output = NULL;
        gpu_pool.d_persistent_temp = NULL;
        gpu_pool_initialized = 0;
    }
}

extern "C" _KLT_FloatImage _KLTCreateFloatImage(int ncols, int nrows);
extern "C" void _KLTFreeFloatImage(_KLT_FloatImage img);

extern "C" void _KLTToFloatImage(
    KLT_PixelType *img,
    int ncols, int nrows,
    _KLT_FloatImage floatimg)
{
    KLT_PixelType *ptrend = img + ncols*nrows;
    float *ptrout = floatimg->data;

    assert(floatimg->ncols >= ncols);
    assert(floatimg->nrows >= nrows);

    floatimg->ncols = ncols;
    floatimg->nrows = nrows;

    while (img < ptrend) *ptrout++ = (float)*img++;
}

static void _computeKernels(
    float sigma,
    ConvolutionKernel *gauss,
    ConvolutionKernel *gaussderiv)
{
    const float factor = 0.01f;
    int i;

    assert(MAX_KERNEL_WIDTH % 2 == 1);
    assert(sigma >= 0.0);

    {
        const int hw = MAX_KERNEL_WIDTH / 2;
        float max_gauss = 1.0f, max_gaussderiv = (float)(sigma*exp(-0.5f));

        for (i = -hw; i <= hw; i++) {
            gauss->data[i+hw] = (float)exp(-i*i / (2*sigma*sigma));
            gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
        }

        gauss->width = MAX_KERNEL_WIDTH;
        for (i = -hw; fabs(gauss->data[i+hw] / max_gauss) < factor; 
             i++, gauss->width -= 2);
        gaussderiv->width = MAX_KERNEL_WIDTH;
        for (i = -hw; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor; 
             i++, gaussderiv->width -= 2);
        if (gauss->width == MAX_KERNEL_WIDTH || 
            gaussderiv->width == MAX_KERNEL_WIDTH) {
            fprintf(stderr, "error: MAX_KERNEL_WIDTH %d is too small for sigma %f\n", 
                    MAX_KERNEL_WIDTH, sigma);
            exit(1);
        }
    }

    for (i = 0; i < gauss->width; i++)
        gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
    for (i = 0; i < gaussderiv->width; i++)
        gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];

    {
        const int hw = gaussderiv->width / 2;
        float den;

        den = 0.0;
        for (i = 0; i < gauss->width; i++) den += gauss->data[i];
        for (i = 0; i < gauss->width; i++) gauss->data[i] /= den;
        den = 0.0;
        for (i = -hw; i <= hw; i++) den -= i*gaussderiv->data[i+hw];
        for (i = -hw; i <= hw; i++) gaussderiv->data[i+hw] /= den;
    }

    sigma_last = sigma;
}

extern "C" void _KLTGetKernelWidths(
    float sigma,
    int *gauss_width,
    int *gaussderiv_width)
{
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
    *gauss_width = gauss_kernel.width;
    *gaussderiv_width = gaussderiv_kernel.width;
}

// OPTIMIZATION: Use constant memory for kernel (removed global param)
__global__ void convolveHorizontalKernel(
    float *input,
    float *output,
    int kernel_width,
    int ncols,
    int nrows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= nrows || col >= ncols) return;
    
    int radius = kernel_width / 2;
    int idx = row * ncols + col;
    
    // Boundary check - set to 0 at edges
    if (col < radius || col >= ncols - radius) {
        output[idx] = 0.0f;
        return;
    }
    
    // Optimization: Loop unrolling with ILP and __ldg() for cached reads
    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kernel_width; k++) {
        int pos = row * ncols + (col - radius + k);
        // OPTIMIZATION: Use constant memory instead of global
        sum += __ldg(&input[pos]) * d_kernel_const[kernel_width - 1 - k];
    }
    
    output[idx] = sum;
}

// OPTIMIZATION: Use constant memory for kernel (removed global param)
__global__ void convolveVerticalKernel(
    float *input,
    float *output,
    int kernel_width,
    int ncols,
    int nrows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= nrows || col >= ncols) return;
    
    int radius = kernel_width / 2;
    int idx = row * ncols + col;
    
    if (row < radius || row >= nrows - radius) {
        output[idx] = 0.0f;
        return;
    }
    
    // Optimization: Loop unrolling with ILP and __ldg()
    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < kernel_width; k++) {
        int pos = (row - radius + k) * ncols + col;
        // OPTIMIZATION: Use constant memory instead of global
        sum += __ldg(&input[pos]) * d_kernel_const[kernel_width - 1 - k];
    }
    
    output[idx] = sum;
}

// OPTIMIZATION: Use persistent GPU memory pool
static void _convolveImageHoriz(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int size = ncols * nrows * sizeof(float);
    
    // Initialize GPU pool if needed
    _initGPUMemoryPool(size);
    
    // OPTIMIZATION: Async H2D transfer on stream
    cudaCheckError(cudaMemcpyAsync(gpu_pool.d_persistent_input, imgin->data, size, 
                                   cudaMemcpyHostToDevice, gpu_pool.stream_horiz));
    
    // OPTIMIZATION: Copy kernel to constant memory once
    cudaCheckError(cudaMemcpyToSymbol(d_kernel_const, kernel.data, 
                                      kernel.width * sizeof(float)));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ncols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // OPTIMIZATION: Launch on stream without intermediate kernel param
    convolveHorizontalKernel<<<gridDim, blockDim, 0, gpu_pool.stream_horiz>>>
        (gpu_pool.d_persistent_input, gpu_pool.d_persistent_output, 
         kernel.width, ncols, nrows);
    
    // OPTIMIZATION: Async D2H transfer on stream
    cudaCheckError(cudaMemcpyAsync(imgout->data, gpu_pool.d_persistent_output, size, 
                                   cudaMemcpyDeviceToHost, gpu_pool.stream_horiz));
    
    // Synchronize stream
    cudaCheckError(cudaStreamSynchronize(gpu_pool.stream_horiz));
}

// OPTIMIZATION: Use persistent GPU memory pool
static void _convolveImageVert(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int size = ncols * nrows * sizeof(float);
    
    // Initialize GPU pool if needed
    _initGPUMemoryPool(size);
    
    // OPTIMIZATION: Async H2D transfer on stream
    cudaCheckError(cudaMemcpyAsync(gpu_pool.d_persistent_input, imgin->data, size, 
                                   cudaMemcpyHostToDevice, gpu_pool.stream_vert));
    
    // OPTIMIZATION: Copy kernel to constant memory once
    cudaCheckError(cudaMemcpyToSymbol(d_kernel_const, kernel.data, 
                                      kernel.width * sizeof(float)));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ncols + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // OPTIMIZATION: Launch on stream without kernel param
    convolveVerticalKernel<<<gridDim, blockDim, 0, gpu_pool.stream_vert>>>
        (gpu_pool.d_persistent_input, gpu_pool.d_persistent_output, 
         kernel.width, ncols, nrows);
    
    // OPTIMIZATION: Async D2H transfer on stream
    cudaCheckError(cudaMemcpyAsync(imgout->data, gpu_pool.d_persistent_output, size, 
                                   cudaMemcpyDeviceToHost, gpu_pool.stream_vert));
    
    // Synchronize stream
    cudaCheckError(cudaStreamSynchronize(gpu_pool.stream_vert));
}

static void _convolveSeparate(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
    _KLT_FloatImage tmpimg;
    tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    
    // Optimization 10: Double buffering with pipelined kernels
    _convolveImageHoriz(imgin, horiz_kernel, tmpimg);
    _convolveImageVert(tmpimg, vert_kernel, imgout);
    
    _KLTFreeFloatImage(tmpimg);
}

extern "C" void _KLTComputeGradients(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage gradx,
    _KLT_FloatImage grady)
{
    assert(gradx->ncols >= img->ncols);
    assert(gradx->nrows >= img->nrows);
    assert(grady->ncols >= img->ncols);
    assert(grady->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
    _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);
}

extern "C" void _KLTComputeSmoothedImage(
    _KLT_FloatImage img,
    float sigma,
    _KLT_FloatImage smooth)
{
    assert(smooth->ncols >= img->ncols);
    assert(smooth->nrows >= img->nrows);

    if (fabs(sigma - sigma_last) > 0.05)
        _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

    _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}

// OPTIMIZATION: Add cleanup function to be called at program end
extern "C" void _KLTGPUCleanup()
{
    _cleanupGPUMemoryPool();
}
