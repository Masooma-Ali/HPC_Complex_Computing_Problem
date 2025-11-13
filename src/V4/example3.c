/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
sequence of images: frame_0001.pgm to frame_0044.pgm.
The sequential mode is set in order to speed processing.
The features are stored in a feature table and written to files.

V4: OpenACC-ready version with proper data management
-----------------------------------------------------
This version fixes the data coherency issues from earlier OpenACC attempts.
The key architectural decisions:

1. REMOVED problematic copyin/copyout clauses that were:
   - Destroying internal KLT data structures
   - Breaking sequential mode pyramid caching
   - Causing feature tracking to fail completely

2. CPU-based processing maintained for correctness:
   - All internal KLT structures stay on CPU
   - Pointer coherency preserved across function calls
   - Sequential mode caching works correctly

3. GPU infrastructure ready for future optimization:
   - OpenACC runtime initialized for future use
   - Code structure allows targeted GPU acceleration
   - Can add GPU offload for compute-intensive kernels later

PERFORMANCE NOTE: For proper GPU acceleration, would need to:
- Create persistent GPU data regions for ALL KLT structures
- Manage pyramid data on GPU across frames
- Use 'present' clauses instead of copyin/copyout
- Only copy feature results back to CPU

Current version prioritizes CORRECTNESS over GPU speed.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <openacc.h>
#include "pnmio.h"
#include "klt.h"

/* #define REPLACE */

#ifdef WIN32
int RunExample3()
#else
int main()
#endif
{
  unsigned char *img1, *img2;
  char fnamein[200], fnameout[200];
  KLT_TrackingContext tc;
  KLT_FeatureList fl;
  KLT_FeatureTable ft;
  int nFeatures = 500;
  int startFrame = 320, endFrame = 600;
  int nFrames = endFrame - startFrame + 1;
  int ncols, nrows;
  int i, frame;

  // Timing variables
  struct timeval tv_start, tv_stop;
  double total_time_ms = 0.0;

  /* OpenACC: Initialize GPU - for future GPU acceleration */
  acc_init(acc_device_nvidia);
  
  // Create tracking structures
  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);

  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */

  // Read the first image
  sprintf(fnamein, "newset/frame_%06d.pgm", startFrame);
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));

  // ========== START TOTAL TIMING ==========
  gettimeofday(&tv_start, NULL);
  // ========================================

  // Select good features from the first image
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
//   KLTStoreFeatureList(fl, ft, 0);
//   sprintf(fnameout, "feat_%04d.ppm", startFrame);
//   KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);

  // Track features through all subsequent frames
  for (i = 1, frame = startFrame + 1; frame <= endFrame; i++, frame++) {
    sprintf(fnamein, "newset/frame_%06d.pgm", frame);
    pgmReadFile(fnamein, img2, &ncols, &nrows);

    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

   //  KLTStoreFeatureList(fl, ft, i);
   //  sprintf(fnameout, "feat_%04d.ppm", frame);
   //  KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);
  }

  // Save feature table results
//   KLTWriteFeatureTable(ft, "features.txt", "%5.1f");
//   KLTWriteFeatureTable(ft, "features.ft", NULL);

  // ========== STOP TOTAL TIMING ==========
  gettimeofday(&tv_stop, NULL);
  total_time_ms = (tv_stop.tv_sec - tv_start.tv_sec) * 1000.0 +
                  (tv_stop.tv_usec - tv_start.tv_usec) / 1000.0;
  // =======================================

  printf("\n");
  printf("╔══════════════════════════════════════════════════════════════════════╗\n");
  printf("║                 TOTAL EXECUTION TIME (V4 GPU OpenACC)                ║\n");
  printf("╠══════════════════════════════════════════════════════════════════════╣\n");
  printf("║ Total Time (All %d frames + File I/O): %10.3f ms              ║\n", nFrames, total_time_ms);
  printf("║ Average Time per Frame:                 %10.3f ms              ║\n", total_time_ms / nFrames);
  printf("╚══════════════════════════════════════════════════════════════════════╝\n");
  printf("\n");

  // Free memory
  KLTFreeFeatureTable(ft);
  KLTFreeFeatureList(fl);
  KLTFreeTrackingContext(tc);
  free(img1);
  free(img2);
  
  /* OpenACC: Shutdown */
  acc_shutdown(acc_device_nvidia);

  return 0;
}

