/*********************************************************************
 * convolve.c
 *
 * V4 PHASE 2: OpenACC GPU-Accelerated Convolution
 * ------------------------------------------------
 * OPENACC OPTIMIZATIONS APPLIED:
 * - Parallel loops with gang, worker, vector hierarchy
 * - Data regions with explicit copyin/copyout for efficiency
 * - Private clauses for thread-safe operations
 * - Sequential inner loops for kernel application
 * 
 * KEY OPTIMIZATIONS:
 * 1. _convolveImageHoriz: Parallelized across rows (gang/worker/vector)
 * 2. _convolveImageVert: Parallelized across columns (gang/worker/vector)
 * 3. _KLTToFloatImage: Vectorized pixel conversion (256 threads)
 * 4. _computeKernels: CPU execution (small arrays, infrequent calls)
 * 
 * DESIGN DECISIONS:
 * - Kernel computation on CPU: Only 71 elements, called once per sigma change
 * - Convolution on GPU: Large image operations, called 12-24x per frame
 * - Explicit data management to avoid "partially present" errors
 * 
 * EXPECTED PERFORMANCE:
 * - Horizontal/Vertical convolution: 100-500x speedup
 * - Float conversion: 50-100x speedup
 * - Overall frame processing: 10-50x speedup
 *********************************************************************/

/* Standard includes */
#include <assert.h>
#include <math.h>
#include <stdlib.h>   /* malloc(), realloc() */

/* Our includes */
#include "base.h"
#include "error.h"
#include "convolve.h"
#include "klt_util.h"   /* printing */

#define MAX_KERNEL_WIDTH 	71


typedef struct  {
  int width;
  float data[MAX_KERNEL_WIDTH];
}  ConvolutionKernel;

/* Kernels */
static ConvolutionKernel gauss_kernel;
static ConvolutionKernel gaussderiv_kernel;
static float sigma_last = -10.0;


/*********************************************************************
 * _KLTToFloatImage
 *
 * Given a pointer to image data (probably unsigned chars), copy
 * data to a float image.
 */

void _KLTToFloatImage(
  KLT_PixelType *img,
  int ncols, int nrows,
  _KLT_FloatImage floatimg)
{
  int total_pixels = ncols * nrows;
  float *ptrout = floatimg->data;

  /* Output image must be large enough to hold result */
  assert(floatimg->ncols >= ncols);
  assert(floatimg->nrows >= nrows);

  floatimg->ncols = ncols;
  floatimg->nrows = nrows;

  /* OpenACC: Parallelize pixel conversion with optimized data movement */
  #pragma acc parallel loop copyin(img[0:total_pixels]) \
          copyout(ptrout[0:total_pixels]) \
          vector_length(256)
  for (int i = 0; i < total_pixels; i++) {
    ptrout[i] = (float) img[i];
  }
}


/*********************************************************************
 * _computeKernels
 */

static void _computeKernels(
  float sigma,
  ConvolutionKernel *gauss,
  ConvolutionKernel *gaussderiv)
{
  const float factor = 0.01f;   /* for truncating tail */
  int i;

  assert(MAX_KERNEL_WIDTH % 2 == 1);
  assert(sigma >= 0.0);

  /* Compute kernels, and automatically determine widths */
  {
    const int hw = MAX_KERNEL_WIDTH / 2;
    float max_gauss = 1.0f, max_gaussderiv = (float) (sigma*exp(-0.5f));
	
    /* Compute gauss and deriv - CPU execution (small array, infrequent call) */
    for (i = -hw ; i <= hw ; i++)  {
      gauss->data[i+hw]      = (float) exp(-i*i / (2*sigma*sigma));
      gaussderiv->data[i+hw] = -i * gauss->data[i+hw];
    }

    /* Compute widths - sequential (data dependent) */
    gauss->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gauss->data[i+hw] / max_gauss) < factor ; 
         i++, gauss->width -= 2);
    gaussderiv->width = MAX_KERNEL_WIDTH;
    for (i = -hw ; fabs(gaussderiv->data[i+hw] / max_gaussderiv) < factor ; 
         i++, gaussderiv->width -= 2);
    if (gauss->width == MAX_KERNEL_WIDTH || 
        gaussderiv->width == MAX_KERNEL_WIDTH)
      KLTError("(_computeKernels) MAX_KERNEL_WIDTH %d is too small for "
               "a sigma of %f", MAX_KERNEL_WIDTH, sigma);
  }

  /* Shift if width less than MAX_KERNEL_WIDTH */
  for (i = 0 ; i < gauss->width ; i++)
    gauss->data[i] = gauss->data[i+(MAX_KERNEL_WIDTH-gauss->width)/2];
  for (i = 0 ; i < gaussderiv->width ; i++)
    gaussderiv->data[i] = gaussderiv->data[i+(MAX_KERNEL_WIDTH-gaussderiv->width)/2];
  
  /* Normalize gauss and deriv */
  {
    const int hw = gaussderiv->width / 2;
    float den;
		
    den = 0.0;
    for (i = 0 ; i < gauss->width ; i++)  den += gauss->data[i];
    for (i = 0 ; i < gauss->width ; i++)  gauss->data[i] /= den;
    den = 0.0;
    for (i = -hw ; i <= hw ; i++)  den -= i*gaussderiv->data[i+hw];
    for (i = -hw ; i <= hw ; i++)  gaussderiv->data[i+hw] /= den;
  }

  sigma_last = sigma;
}
	

/*********************************************************************
 * _KLTGetKernelWidths
 *
 */

void _KLTGetKernelWidths(
  float sigma,
  int *gauss_width,
  int *gaussderiv_width)
{
  _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
  *gauss_width = gauss_kernel.width;
  *gaussderiv_width = gaussderiv_kernel.width;
}


/*********************************************************************
 * _convolveImageHoriz
 */

static void _convolveImageHoriz(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *imgin_data = imgin->data;
  float *imgout_data = imgout->data;
  int radius = kernel.width / 2;
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* OpenACC: Parallelize horizontal convolution across rows 
   * Note: firstprivate(kernel) copies entire struct to GPU, avoiding "partially present" errors
   * The kernel is small (288 bytes) and read-only, so this is efficient */
  #pragma acc data copyin(imgin_data[0:ncols*nrows]) \
                   copyout(imgout_data[0:ncols*nrows])
  {
    /* Process all rows in parallel */
    #pragma acc parallel loop gang worker firstprivate(kernel, radius)
    for (j = 0 ; j < nrows ; j++)  {
      int row_offset = j * ncols;
      
      /* Zero left-border */
      #pragma acc loop vector
      for (i = 0 ; i < radius ; i++) {
        imgout_data[row_offset + i] = 0.0;
      }

      /* Convolve middle columns with kernel */
      #pragma acc loop vector private(k)
      for (i = radius ; i < ncols - radius ; i++)  {
        float sum = 0.0;
        #pragma acc loop seq
        for (k = 0 ; k < kernel.width ; k++) {
          sum += imgin_data[row_offset + i - radius + k] * kernel.data[kernel.width - 1 - k];
        }
        imgout_data[row_offset + i] = sum;
      }

      /* Zero right-border */ 
      #pragma acc loop vector
      for (i = ncols - radius ; i < ncols ; i++) {
        imgout_data[row_offset + i] = 0.0;
      }
    }
  }
}


/*********************************************************************
 * _convolveImageVert
 */

static void _convolveImageVert(
  _KLT_FloatImage imgin,
  ConvolutionKernel kernel,
  _KLT_FloatImage imgout)
{
  float *imgin_data = imgin->data;
  float *imgout_data = imgout->data;
  int radius = kernel.width / 2;
  int ncols = imgin->ncols, nrows = imgin->nrows;
  int i, j, k;

  /* Kernel width must be odd */
  assert(kernel.width % 2 == 1);

  /* Must read from and write to different images */
  assert(imgin != imgout);

  /* Output image must be large enough to hold result */
  assert(imgout->ncols >= imgin->ncols);
  assert(imgout->nrows >= imgin->nrows);

  /* OpenACC: Parallelize vertical convolution across columns
   * Note: firstprivate(kernel) copies entire struct to GPU, avoiding "partially present" errors
   * The kernel is small (288 bytes) and read-only, so this is efficient */
  #pragma acc data copyin(imgin_data[0:ncols*nrows]) \
                   copyout(imgout_data[0:ncols*nrows])
  {
    /* Process all columns in parallel */
    #pragma acc parallel loop gang worker firstprivate(kernel, radius)
    for (i = 0 ; i < ncols ; i++)  {

      /* Zero top-border */
      #pragma acc loop vector
      for (j = 0 ; j < radius ; j++)  {
        imgout_data[j * ncols + i] = 0.0;
      }

      /* Convolve middle rows with kernel */
      #pragma acc loop vector private(k)
      for (j = radius ; j < nrows - radius ; j++)  {
        float sum = 0.0;
        #pragma acc loop seq
        for (k = 0 ; k < kernel.width ; k++)  {
          sum += imgin_data[(j - radius + k) * ncols + i] * kernel.data[kernel.width - 1 - k];
        }
        imgout_data[j * ncols + i] = sum;
      }

      /* Zero bottom-border */
      #pragma acc loop vector
      for (j = nrows - radius ; j < nrows ; j++)  {
        imgout_data[j * ncols + i] = 0.0;
      }
    }
  }
}


/*********************************************************************
 * _convolveSeparate
 */

static void _convolveSeparate(
  _KLT_FloatImage imgin,
  ConvolutionKernel horiz_kernel,
  ConvolutionKernel vert_kernel,
  _KLT_FloatImage imgout)
{
  /* Create temporary image */
  _KLT_FloatImage tmpimg;
  tmpimg = _KLTCreateFloatImage(imgin->ncols, imgin->nrows);
  
  /* Do convolution */
  _convolveImageHoriz(imgin, horiz_kernel, tmpimg);

  _convolveImageVert(tmpimg, vert_kernel, imgout);

  /* Free memory */
  _KLTFreeFloatImage(tmpimg);
}

	
/*********************************************************************
 * _KLTComputeGradients
 */

void _KLTComputeGradients(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage gradx,
  _KLT_FloatImage grady)
{
				
  /* Output images must be large enough to hold result */
  assert(gradx->ncols >= img->ncols);
  assert(gradx->nrows >= img->nrows);
  assert(grady->ncols >= img->ncols);
  assert(grady->nrows >= img->nrows);

  /* Compute kernels, if necessary */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);
	
  _convolveSeparate(img, gaussderiv_kernel, gauss_kernel, gradx);
  _convolveSeparate(img, gauss_kernel, gaussderiv_kernel, grady);

}
	

/*********************************************************************
 * _KLTComputeSmoothedImage
 */

void _KLTComputeSmoothedImage(
  _KLT_FloatImage img,
  float sigma,
  _KLT_FloatImage smooth)
{
  /* Output image must be large enough to hold result */
  assert(smooth->ncols >= img->ncols);
  assert(smooth->nrows >= img->nrows);

  /* Compute kernel, if necessary; gauss_deriv is not used */
  if (fabs(sigma - sigma_last) > 0.05)
    _computeKernels(sigma, &gauss_kernel, &gaussderiv_kernel);

  _convolveSeparate(img, gauss_kernel, gauss_kernel, smooth);
}



