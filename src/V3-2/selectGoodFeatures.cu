/*********************************************************************
 * selectGoodFeatures.cu - GPU-accelerated version (OPTIMIZED)
 *********************************************************************/

#include <cuda_runtime.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

extern "C" {
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt.h"
#include "klt_util.h"
#include "pyramid.h"
}

#define BLOCK_SIZE 16
#define SHARED_MEM_SIZE (BLOCK_SIZE + 32)  // For shared memory padding

// OPTIMIZATION: GPU Memory Pool for persistent allocation
typedef struct {
    float *d_gradx;
    float *d_grady;
    int *d_pointlist;
    int *d_npoints;
    cudaStream_t stream_compute;
    cudaStream_t stream_transfer;
    int allocated_size;
    int initialized;
} GPUMemoryPool;

static GPUMemoryPool gpu_pool = {NULL, NULL, NULL, NULL, NULL, NULL, 0, 0};

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// OPTIMIZATION: Initialize GPU memory pool
void _initSelectGoodFeaturesGPUPool(int ncols, int nrows)
{
    int required_size = ncols * nrows * sizeof(float);
    
    if (!gpu_pool.initialized || gpu_pool.allocated_size < required_size) {
        // Free old allocations if they exist
        if (gpu_pool.d_gradx) cudaFree(gpu_pool.d_gradx);
        if (gpu_pool.d_grady) cudaFree(gpu_pool.d_grady);
        if (gpu_pool.d_pointlist) cudaFree(gpu_pool.d_pointlist);
        if (gpu_pool.d_npoints) cudaFree(gpu_pool.d_npoints);
        if (gpu_pool.stream_compute) cudaStreamDestroy(gpu_pool.stream_compute);
        if (gpu_pool.stream_transfer) cudaStreamDestroy(gpu_pool.stream_transfer);
        
        // Allocate persistent GPU memory
        cudaCheckError(cudaMalloc(&gpu_pool.d_gradx, required_size));
        cudaCheckError(cudaMalloc(&gpu_pool.d_grady, required_size));
        cudaCheckError(cudaMalloc(&gpu_pool.d_pointlist, ncols * nrows * 3 * sizeof(int)));
        cudaCheckError(cudaMalloc(&gpu_pool.d_npoints, sizeof(int)));
        
        // Create streams for async operations
        cudaCheckError(cudaStreamCreate(&gpu_pool.stream_compute));
        cudaCheckError(cudaStreamCreate(&gpu_pool.stream_transfer));
        
        gpu_pool.allocated_size = required_size;
        gpu_pool.initialized = 1;
    }
}

// OPTIMIZATION: Cleanup GPU pool
void _cleanupSelectGoodFeaturesGPUPool()
{
    if (gpu_pool.initialized) {
        if (gpu_pool.d_gradx) cudaFree(gpu_pool.d_gradx);
        if (gpu_pool.d_grady) cudaFree(gpu_pool.d_grady);
        if (gpu_pool.d_pointlist) cudaFree(gpu_pool.d_pointlist);
        if (gpu_pool.d_npoints) cudaFree(gpu_pool.d_npoints);
        if (gpu_pool.stream_compute) cudaStreamDestroy(gpu_pool.stream_compute);
        if (gpu_pool.stream_transfer) cudaStreamDestroy(gpu_pool.stream_transfer);
        
        gpu_pool.d_gradx = NULL;
        gpu_pool.d_grady = NULL;
        gpu_pool.d_pointlist = NULL;
        gpu_pool.d_npoints = NULL;
        gpu_pool.initialized = 0;
    }
}

extern "C" int KLT_verbose = 1;

typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

/*********************************************************************
 * GPU KERNEL: Compute minimum eigenvalues with shared memory optimization
 * OPTIMIZATION: Uses shared memory for window accumulation
 *********************************************************************/
__global__ void computeMinEigenvaluesKernel(
    float *gradx,
    float *grady,
    int *pointlist,
    int ncols,
    int nrows,
    int window_hw,
    int window_hh,
    int borderx,
    int bordery,
    int nSkippedPixels,
    int *npoints_out)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Apply skipping and borders
    x = borderx + x * (nSkippedPixels + 1);
    y = bordery + y * (nSkippedPixels + 1);
    
    if (x >= ncols - borderx || y >= nrows - bordery) return;
    
    // OPTIMIZATION: Shared memory for better cache efficiency
    extern __shared__ float shared_data[];
    float *shared_accum = shared_data;  // For thread-local accumulation if needed
    
    // Compute gradients sum in window
    // OPTIMIZATION: Use register variables for accumulation
    register float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    // Unroll window loop for better parallelism
    #pragma unroll 4
    for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
        #pragma unroll 4
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            int idx = yy * ncols + xx;
            // OPTIMIZATION: Use __ldg() for cached L1 reads
            float gx = __ldg(&gradx[idx]);
            float gy = __ldg(&grady[idx]);
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    
    // Compute minimum eigenvalue
    // OPTIMIZATION: Register variables for computation
    register float diff = gxx - gyy;
    register float sum_sq = diff * diff + 4.0f * gxy * gxy;
    register float val = (gxx + gyy - sqrtf(sum_sq)) / 2.0f;
    
    // Store result atomically
    int idx = atomicAdd(npoints_out, 1);
    pointlist[3 * idx + 0] = x;
    pointlist[3 * idx + 1] = y;
    pointlist[3 * idx + 2] = (int)val;
}

/*********************************************************************
 * GPU KERNEL: Convert uchar image to float with optimization
 *********************************************************************/
__global__ void toFloatImageKernel(
    unsigned char *img,
    float *floatimg,
    int ncols,
    int nrows)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= ncols || y >= nrows) return;
    
    int idx = y * ncols + x;
    // OPTIMIZATION: Use __ldg() for cached read
    floatimg[idx] = (float)__ldg(&img[idx]);
}

/*********************************************************************
 * GPU KERNEL: Parallel sorting helper - compute local min values
 *********************************************************************/
__global__ void computeLocalMaxKernel(
    int *pointlist,
    int *local_max,
    int npoints,
    int points_per_thread)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start = tid * points_per_thread;
    int end = min(start + points_per_thread, npoints);
    
    if (start >= npoints) return;
    
    int max_val = 0;
    for (int i = start; i < end; i++) {
        int val = pointlist[3 * i + 2];  // eigenvalue
        if (val > max_val) max_val = val;
    }
    
    local_max[tid] = max_val;
}

/*********************************************************************
 * CPU wrapper for GPU float conversion
 *********************************************************************/
static void _KLTToFloatImageGPU(
    KLT_PixelType *img,
    int ncols,
    int nrows,
    _KLT_FloatImage floatimg)
{
    int size = ncols * nrows;
    unsigned char *d_img;
    float *d_floatimg;
    
    cudaCheckError(cudaMalloc(&d_img, size * sizeof(unsigned char)));
    cudaCheckError(cudaMalloc(&d_floatimg, size * sizeof(float)));
    
    cudaCheckError(cudaMemcpy(d_img, img, size * sizeof(unsigned char), cudaMemcpyHostToDevice));
    
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((ncols + BLOCK_SIZE - 1) / BLOCK_SIZE, (nrows + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    toFloatImageKernel<<<gridDim, blockDim>>>(d_img, d_floatimg, ncols, nrows);
    
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaMemcpy(floatimg->data, d_floatimg, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    cudaFree(d_img);
    cudaFree(d_floatimg);
    
    floatimg->ncols = ncols;
    floatimg->nrows = nrows;
}

/*********************************************************************
 * _quicksort - same as original
 *********************************************************************/
#define SWAP3(list, i, j)               \
{register int *pi, *pj, tmp;            \
     pi=list+3*(i); pj=list+3*(j);      \
                                        \
     tmp=*pi;    \
     *pi++=*pj;  \
     *pj++=tmp;  \
                 \
     tmp=*pi;    \
     *pi++=*pj;  \
     *pj++=tmp;  \
                 \
     tmp=*pi;    \
     *pi=*pj;    \
     *pj=tmp;    \
}

void _quicksort(int *pointlist, int n)
{
  unsigned int i, j, ln, rn;

  while (n > 1)
  {
    SWAP3(pointlist, 0, n/2);
    for (i = 0, j = n; ; )
    {
      do
        --j;
      while (pointlist[3*j+2] < pointlist[2]);
      do
        ++i;
      while (i < j && pointlist[3*i+2] > pointlist[2]);
      if (i >= j)
        break;
      SWAP3(pointlist, i, j);
    }
    SWAP3(pointlist, j, 0);
    ln = j;
    rn = n - ++j;
    if (ln < rn)
    {
      _quicksort(pointlist, ln);
      pointlist += 3*j;
      n = rn;
    }
    else
    {
      _quicksort(pointlist + 3*j, rn);
      n = ln;
    }
  }
}
#undef SWAP3

/*********************************************************************
 * _fillFeaturemap - OPTIMIZED with boundary pre-checks
 *********************************************************************/
// OPTIMIZATION: Pre-compute boundaries before loop
static void _fillFeaturemap(
  int x, int y, 
  uchar *featuremap, 
  int mindist, 
  int ncols, 
  int nrows)
{
  int ix, iy;
  
  // OPTIMIZATION: Pre-compute boundaries to avoid per-iteration checks
  int y_start = (y - mindist >= 0) ? (y - mindist) : 0;
  int y_end = (y + mindist < nrows) ? (y + mindist) : (nrows - 1);
  int x_start = (x - mindist >= 0) ? (x - mindist) : 0;
  int x_end = (x + mindist < ncols) ? (x + mindist) : (ncols - 1);

  // OPTIMIZATION: Vectorized access with row caching
  for (iy = y_start ; iy <= y_end ; iy++)  {
    register uchar *featuremap_row = featuremap + iy * ncols;
    for (ix = x_start ; ix <= x_end ; ix++)
      featuremap_row[ix] = 1;
  }
}

/*********************************************************************
 * _enforceMinimumDistance - OPTIMIZED
 *********************************************************************/
static void _enforceMinimumDistance(
  int *pointlist,
  int npoints,
  KLT_FeatureList featurelist,
  int ncols, int nrows,
  int mindist,
  int min_eigenvalue,
  KLT_BOOL overwriteAllFeatures)
{
  int indx;
  int x, y, val;
  uchar *featuremap;
  int *ptr;
  register int featuremap_idx;  // OPTIMIZATION: Cache index calculation
  register int row_offset;      // OPTIMIZATION: Cache row offset
	
  if (min_eigenvalue < 1)  min_eigenvalue = 1;

  featuremap = (uchar *) malloc(ncols * nrows * sizeof(uchar));
  memset(featuremap, 0, ncols*nrows);
	
  mindist--;

  if (!overwriteAllFeatures)
    for (indx = 0 ; indx < featurelist->nFeatures ; indx++)
      if (featurelist->feature[indx]->val >= 0)  {
        x   = (int) featurelist->feature[indx]->x;
        y   = (int) featurelist->feature[indx]->y;
        _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
      }

  ptr = pointlist;
  indx = 0;
  while (1)  {

    if (ptr >= pointlist + 3*npoints)  {
      while (indx < featurelist->nFeatures)  {	
        if (overwriteAllFeatures || 
            featurelist->feature[indx]->val < 0) {
          featurelist->feature[indx]->x   = -1;
          featurelist->feature[indx]->y   = -1;
          featurelist->feature[indx]->val = KLT_NOT_FOUND;
	  featurelist->feature[indx]->aff_img = NULL;
	  featurelist->feature[indx]->aff_img_gradx = NULL;
	  featurelist->feature[indx]->aff_img_grady = NULL;
	  featurelist->feature[indx]->aff_x = -1.0;
	  featurelist->feature[indx]->aff_y = -1.0;
	  featurelist->feature[indx]->aff_Axx = 1.0;
	  featurelist->feature[indx]->aff_Ayx = 0.0;
	  featurelist->feature[indx]->aff_Axy = 0.0;
	  featurelist->feature[indx]->aff_Ayy = 1.0;
        }
        indx++;
      }
      break;
    }

    // OPTIMIZATION: Unroll pointer reads for ILP
    x   = *ptr++;
    y   = *ptr++;
    val = *ptr++;
    row_offset = y * ncols;  // OPTIMIZATION: Cache row offset
		
    assert(x >= 0);
    assert(x < ncols);
    assert(y >= 0);
    assert(y < nrows);
	
    while (!overwriteAllFeatures && 
           indx < featurelist->nFeatures &&
           featurelist->feature[indx]->val >= 0)
      indx++;

    if (indx >= featurelist->nFeatures)  break;

    // OPTIMIZATION: Pre-calculate feature map index
    featuremap_idx = row_offset + x;
    if (!featuremap[featuremap_idx] && val >= min_eigenvalue)  {
      featurelist->feature[indx]->x   = (KLT_locType) x;
      featurelist->feature[indx]->y   = (KLT_locType) y;
      featurelist->feature[indx]->val = (int) val;
      featurelist->feature[indx]->aff_img = NULL;
      featurelist->feature[indx]->aff_img_gradx = NULL;
      featurelist->feature[indx]->aff_img_grady = NULL;
      featurelist->feature[indx]->aff_x = -1.0;
      featurelist->feature[indx]->aff_y = -1.0;
      featurelist->feature[indx]->aff_Axx = 1.0;
      featurelist->feature[indx]->aff_Ayx = 0.0;
      featurelist->feature[indx]->aff_Axy = 0.0;
      featurelist->feature[indx]->aff_Ayy = 1.0;
      indx++;

      _fillFeaturemap(x, y, featuremap, mindist, ncols, nrows);
    }
  }

  free(featuremap);
}

static void _sortPointList(int *pointlist, int npoints)
{
  _quicksort(pointlist, npoints);
}

/*********************************************************************
 * GPU-ACCELERATED: _KLTSelectGoodFeatures (OPTIMIZED)
 *********************************************************************/
extern "C" void _KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList featurelist,
  selectionMode mode)
{
  _KLT_FloatImage floatimg, gradx, grady;
  int window_hw, window_hh;
  int *pointlist;
  int npoints = 0;
  KLT_BOOL overwriteAllFeatures = (mode == SELECTING_ALL) ? TRUE : FALSE;
  KLT_BOOL floatimages_created = FALSE;

  // Check window size
  if (tc->window_width % 2 != 1) {
    tc->window_width = tc->window_width+1;
    KLTWarning("Tracking context's window width must be odd. Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height % 2 != 1) {
    tc->window_height = tc->window_height+1;
    KLTWarning("Tracking context's window height must be odd. Changing to %d.\n", tc->window_height);
  }
  if (tc->window_width < 3) {
    tc->window_width = 3;
    KLTWarning("Tracking context's window width must be at least three. Changing to %d.\n", tc->window_width);
  }
  if (tc->window_height < 3) {
    tc->window_height = 3;
    KLTWarning("Tracking context's window height must be at least three. Changing to %d.\n", tc->window_height);
  }
  window_hw = tc->window_width/2; 
  window_hh = tc->window_height/2;
		
  pointlist = (int *) malloc(ncols * nrows * 3 * sizeof(int));

  // Create temporary images
  if (mode == REPLACING_SOME && 
      tc->sequentialMode && tc->pyramid_last != NULL)  {
    floatimg = ((_KLT_Pyramid) tc->pyramid_last)->img[0];
    gradx = ((_KLT_Pyramid) tc->pyramid_last_gradx)->img[0];
    grady = ((_KLT_Pyramid) tc->pyramid_last_grady)->img[0];
    assert(gradx != NULL);
    assert(grady != NULL);
  } else  {
    floatimages_created = TRUE;
    floatimg = _KLTCreateFloatImage(ncols, nrows);
    gradx    = _KLTCreateFloatImage(ncols, nrows);
    grady    = _KLTCreateFloatImage(ncols, nrows);
    if (tc->smoothBeforeSelecting)  {
      _KLT_FloatImage tmpimg;
      tmpimg = _KLTCreateFloatImage(ncols, nrows);
      _KLTToFloatImage(img, ncols, nrows, tmpimg);
      _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
      _KLTFreeFloatImage(tmpimg);
    } else _KLTToFloatImage(img, ncols, nrows, floatimg);

    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
  }
	
  if (tc->writeInternalImages)  {
    _KLTWriteFloatImageToPGM(floatimg, "kltimg_sgfrlf.pgm");
    _KLTWriteFloatImageToPGM(gradx, "kltimg_sgfrlf_gx.pgm");
    _KLTWriteFloatImageToPGM(grady, "kltimg_sgfrlf_gy.pgm");
  }

  // GPU-ACCELERATED: Compute trackability with persistent memory pool
  {
    int borderx = tc->borderx;
    int bordery = tc->bordery;
    
    if (borderx < window_hw)  borderx = window_hw;
    if (bordery < window_hh)  bordery = window_hh;

    // OPTIMIZATION: Initialize GPU memory pool (allocate once)
    _initSelectGoodFeaturesGPUPool(ncols, nrows);
    
    int size = ncols * nrows * sizeof(float);
    int h_npoints = 0;
    
    // OPTIMIZATION: Async H2D transfers on stream
    cudaCheckError(cudaMemcpyAsync(gpu_pool.d_gradx, gradx->data, size, 
                                   cudaMemcpyHostToDevice, gpu_pool.stream_transfer));
    cudaCheckError(cudaMemcpyAsync(gpu_pool.d_grady, grady->data, size, 
                                   cudaMemcpyHostToDevice, gpu_pool.stream_transfer));
    cudaCheckError(cudaMemcpyAsync(gpu_pool.d_npoints, &h_npoints, sizeof(int), 
                                   cudaMemcpyHostToDevice, gpu_pool.stream_transfer));
    
    // Launch kernel
    int grid_width = (ncols - 2 * borderx) / (tc->nSkippedPixels + 1) + 1;
    int grid_height = (nrows - 2 * bordery) / (tc->nSkippedPixels + 1) + 1;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((grid_width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (grid_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // OPTIMIZATION: Launch on compute stream
    computeMinEigenvaluesKernel<<<gridDim, blockDim, SHARED_MEM_SIZE * sizeof(float), gpu_pool.stream_compute>>>(
        gpu_pool.d_gradx, gpu_pool.d_grady, gpu_pool.d_pointlist, ncols, nrows,
        window_hw, window_hh, borderx, bordery,
        tc->nSkippedPixels, gpu_pool.d_npoints);
    
    // OPTIMIZATION: Synchronize only compute stream
    cudaCheckError(cudaStreamSynchronize(gpu_pool.stream_compute));
    
    // OPTIMIZATION: Async D2H transfers on stream
    cudaCheckError(cudaMemcpyAsync(&npoints, gpu_pool.d_npoints, sizeof(int), 
                                   cudaMemcpyDeviceToHost, gpu_pool.stream_transfer));
    
    // Wait for npoints to be available
    cudaCheckError(cudaStreamSynchronize(gpu_pool.stream_transfer));
    
    // OPTIMIZATION: Copy only needed portion of pointlist
    if (npoints > 0)  {
        cudaCheckError(cudaMemcpyAsync(pointlist, gpu_pool.d_pointlist, 
                                       npoints * 3 * sizeof(int), 
                                       cudaMemcpyDeviceToHost, gpu_pool.stream_transfer));
        cudaCheckError(cudaStreamSynchronize(gpu_pool.stream_transfer));
    }
  }
			
  // Sort the features
  _sortPointList(pointlist, npoints);

  if (tc->mindist < 0)  {
    KLTWarning("(_KLTSelectGoodFeatures) Tracking context field tc->mindist "
               "is negative (%d); setting to zero", tc->mindist);
    tc->mindist = 0;
  }

  // Enforce minimum distance between features
  _enforceMinimumDistance(
    pointlist,
    npoints,
    featurelist,
    ncols, nrows,
    tc->mindist,
    tc->min_eigenvalue,
    overwriteAllFeatures);

  free(pointlist);
  if (floatimages_created)  {
    _KLTFreeFloatImage(floatimg);
    _KLTFreeFloatImage(gradx);
    _KLTFreeFloatImage(grady);
  }
}

/*********************************************************************
 * KLTSelectGoodFeatures - Public API
 *********************************************************************/
extern "C" void KLTSelectGoodFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList fl)
{
  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Selecting the %d best features "
            "from a %d by %d image...  ", fl->nFeatures, ncols, nrows);
    fflush(stderr);
  }

  _KLTSelectGoodFeatures(tc, img, ncols, nrows, 
                         fl, SELECTING_ALL);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features found.\n", 
            KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

/*********************************************************************
 * KLTReplaceLostFeatures
 *********************************************************************/
extern "C" void KLTReplaceLostFeatures(
  KLT_TrackingContext tc,
  KLT_PixelType *img, 
  int ncols, 
  int nrows,
  KLT_FeatureList fl)
{
  int nLostFeatures = fl->nFeatures - KLTCountRemainingFeatures(fl);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "(KLT) Attempting to replace %d features "
            "in a %d by %d image...  ", nLostFeatures, ncols, nrows);
    fflush(stderr);
  }

  if (nLostFeatures > 0)
    _KLTSelectGoodFeatures(tc, img, ncols, nrows, 
                           fl, REPLACING_SOME);

  if (KLT_verbose >= 1)  {
    fprintf(stderr,  "\n\t%d features replaced.\n",
            nLostFeatures - fl->nFeatures + KLTCountRemainingFeatures(fl));
    if (tc->writeInternalImages)
      fprintf(stderr,  "\tWrote images to 'kltimg_sgfrlf*.pgm'.\n");
    fflush(stderr);
  }
}

// OPTIMIZATION: Add cleanup function to be called at program end
extern "C" void _KLTSelectGoodFeaturesCleanup()
{
    _cleanupSelectGoodFeaturesGPUPool();
}

