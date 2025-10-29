/*********************************************************************
 * selectGoodFeatures.cu - GPU-accelerated version
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
#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "cuda error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

extern "C" int KLT_verbose = 1;

typedef enum {SELECTING_ALL, REPLACING_SOME} selectionMode;

/*********************************************************************
 * GPU KERNEL: Compute minimum eigenvalues for all pixels
 * This is the most computationally intensive part
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
    
    // Compute gradients sum in window - OPTIMIZED WITH __ldg()
    float gxx = 0.0f, gxy = 0.0f, gyy = 0.0f;
    
    // Unroll window loop for better parallelism indication
    #pragma unroll 4
    for (int yy = y - window_hh; yy <= y + window_hh; yy++) {
        #pragma unroll 4
        for (int xx = x - window_hw; xx <= x + window_hw; xx++) {
            int idx = yy * ncols + xx;
            // Use __ldg() for cached L1 reads on gradient images
            float gx = __ldg(&gradx[idx]);
            float gy = __ldg(&grady[idx]);
            gxx += gx * gx;
            gxy += gx * gy;
            gyy += gy * gy;
        }
    }
    
    // Compute minimum eigenvalue
    float val = (gxx + gyy - sqrtf((gxx - gyy) * (gxx - gyy) + 4.0f * gxy * gxy)) / 2.0f;
    
    // Store result atomically
    int idx = atomicAdd(npoints_out, 1);
    pointlist[3 * idx + 0] = x;
    pointlist[3 * idx + 1] = y;
    pointlist[3 * idx + 2] = (int)val;
}

/*********************************************************************
 * GPU KERNEL: Convert uchar image to float (faster than CPU)
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
    // Use __ldg() for cached read of input image
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
 * Helper functions - same as original
 *********************************************************************/
static void _fillFeaturemap(
  int x, int y, 
  uchar *featuremap, 
  int mindist, 
  int ncols, 
  int nrows)
{
  int ix, iy;

  for (iy = y - mindist ; iy <= y + mindist ; iy++)
    for (ix = x - mindist ; ix <= x + mindist ; ix++)
      if (ix >= 0 && ix < ncols && iy >= 0 && iy < nrows)
        featuremap[iy*ncols+ix] = 1;
}

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

    x   = *ptr++;
    y   = *ptr++;
    val = *ptr++;
		
    assert(x >= 0);
    assert(x < ncols);
    assert(y >= 0);
    assert(y < nrows);
	
    while (!overwriteAllFeatures && 
           indx < featurelist->nFeatures &&
           featurelist->feature[indx]->val >= 0)
      indx++;

    if (indx >= featurelist->nFeatures)  break;

    if (!featuremap[y*ncols+x] && val >= min_eigenvalue)  {
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
 * GPU-ACCELERATED: _KLTSelectGoodFeatures
 * Main computation routine with GPU acceleration
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
      
      // GPU-ACCELERATED: Float image conversion
      _KLTToFloatImageGPU(img, ncols, nrows, tmpimg);
      
      // Already GPU-accelerated in convolve.cu
      _KLTComputeSmoothedImage(tmpimg, _KLTComputeSmoothSigma(tc), floatimg);
      _KLTFreeFloatImage(tmpimg);
    } else {
      // GPU-ACCELERATED: Float image conversion
      _KLTToFloatImageGPU(img, ncols, nrows, floatimg);
    }

    // Already GPU-accelerated in convolve.cu
    _KLTComputeGradients(floatimg, tc->grad_sigma, gradx, grady);
  }
	
  if (tc->writeInternalImages)  {
    _KLTWriteFloatImageToPGM(floatimg, "kltimg_sgfrlf.pgm");
    _KLTWriteFloatImageToPGM(gradx, "kltimg_sgfrlf_gx.pgm");
    _KLTWriteFloatImageToPGM(grady, "kltimg_sgfrlf_gy.pgm");
  }

  // GPU-ACCELERATED: Compute trackability (minimum eigenvalues)
  {
    int borderx = tc->borderx;
    int bordery = tc->bordery;
    
    if (borderx < window_hw)  borderx = window_hw;
    if (bordery < window_hh)  bordery = window_hh;

    // Allocate GPU memory
    float *d_gradx, *d_grady;
    int *d_pointlist, *d_npoints;
    int size = ncols * nrows * sizeof(float);
    int h_npoints = 0;
    
    cudaCheckError(cudaMalloc(&d_gradx, size));
    cudaCheckError(cudaMalloc(&d_grady, size));
    cudaCheckError(cudaMalloc(&d_pointlist, ncols * nrows * 3 * sizeof(int)));
    cudaCheckError(cudaMalloc(&d_npoints, sizeof(int)));
    
    cudaCheckError(cudaMemcpy(d_gradx, gradx->data, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_grady, grady->data, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_npoints, &h_npoints, sizeof(int), cudaMemcpyHostToDevice));
    
    // Launch kernel
    int grid_width = (ncols - 2 * borderx) / (tc->nSkippedPixels + 1) + 1;
    int grid_height = (nrows - 2 * bordery) / (tc->nSkippedPixels + 1) + 1;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((grid_width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (grid_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    computeMinEigenvaluesKernel<<<gridDim, blockDim>>>(
        d_gradx, d_grady, d_pointlist, ncols, nrows,
        window_hw, window_hh, borderx, bordery,
        tc->nSkippedPixels, d_npoints);
    
    cudaCheckError(cudaDeviceSynchronize());
    
    // Copy results back
    cudaCheckError(cudaMemcpy(&npoints, d_npoints, sizeof(int), cudaMemcpyDeviceToHost));
    cudaCheckError(cudaMemcpy(pointlist, d_pointlist, npoints * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    
    cudaFree(d_gradx);
    cudaFree(d_grady);
    cudaFree(d_pointlist);
    cudaFree(d_npoints);
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

