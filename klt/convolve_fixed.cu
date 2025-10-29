#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

extern "C" {
#include "base.h"
#include "klt_util.h"
}

#define MAX_KERNEL_WIDTH 71
#define BLOCK_SIZE 16

typedef struct {
    int width;
    float data[MAX_KERNEL_WIDTH];
} ConvolutionKernel;

static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
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

__global__ void convolveHorizontalKernel(
    float *input,
    float *output,
    float *kernel,
    int kernel_width,
    int ncols,
    int nrows)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row >= nrows || col >= ncols) return;
    
    int radius = kernel_width / 2;
    int idx = row * ncols + col;
    
    if (col < radius || col >= ncols - radius) {
        output[idx] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        int pos = row * ncols + (col - radius + k);
        sum += input[pos] * kernel[kernel_width - 1 - k];
    }
    
    output[idx] = sum;
}

__global__ void convolveVerticalKernel(
    float *input,
    float *output,
    float *kernel,
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
    
    float sum = 0.0f;
    for (int k = 0; k < kernel_width; k++) {
        int pos = (row - radius + k) * ncols + col;
        sum += input[pos] * kernel[kernel_width - 1 - k];
    }
    
    output[idx] = sum;
}

static void _convolveImageHoriz(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int size = ncols * nrows * sizeof(float);
    
    float *d_input, *d_output, *d_kernel;
    
    cudaCheckError(cudaMalloc(&d_input, size));
    cudaCheckError(cudaMalloc(&d_output, size));
    cudaCheckError(cudaMalloc(&d_kernel, kernel.width * sizeof(float)));
    
    cudaCheckError(cudaMemcpy(d_input, imgin->data, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_kernel, kernel.data, kernel.width * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ncols + BLOCK_SIZE - 1) / BLOCK_SIZE, (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    convolveHorizontalKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, kernel.width, ncols, nrows);
    
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(imgout->data, d_output, size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

static void _convolveImageVert(
    _KLT_FloatImage imgin,
    ConvolutionKernel kernel,
    _KLT_FloatImage imgout)
{
    int ncols = imgin->ncols;
    int nrows = imgin->nrows;
    int size = ncols * nrows * sizeof(float);
    
    float *d_input, *d_output, *d_kernel;
    
    cudaCheckError(cudaMalloc(&d_input, size));
    cudaCheckError(cudaMalloc(&d_output, size));
    cudaCheckError(cudaMalloc(&d_kernel, kernel.width * sizeof(float)));
    
    cudaCheckError(cudaMemcpy(d_input, imgin->data, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_kernel, kernel.data, kernel.width * sizeof(float), cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ncols + BLOCK_SIZE - 1) / BLOCK_SIZE, (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    convolveVerticalKernel<<<gridDim, blockDim>>>(d_input, d_output, d_kernel, kernel.width, ncols, nrows);
    
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(imgout->data, d_output, size, cudaMemcpyDeviceToHost));
    
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
}

static void _convolveSeparate(
    _KLT_FloatImage imgin,
    ConvolutionKernel horiz_kernel,
    ConvolutionKernel vert_kernel,
    _KLT_FloatImage imgout)
{
    _KLT_FloatImage tmpimg;
    tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
    
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
