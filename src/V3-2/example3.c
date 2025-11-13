/**********************************************************************
Finds the 150 best features in an image and tracks them through the 
sequence of images: frame_0001.pgm to frame_0044.pgm.
The sequential mode is set in order to speed processing.
The features are stored in a feature table and written to files.
**********************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
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
  int startFrame = 320, endFrame = 600;   // <-- updated frame range
  int nFrames = endFrame - startFrame + 1;
  int ncols, nrows;
  int i, frame;

  // Timing variables
  struct timeval tv_start, tv_stop;
  double total_time_ms = 0.0;

  // Create tracking structures
  tc = KLTCreateTrackingContext();
  fl = KLTCreateFeatureList(nFeatures);
  ft = KLTCreateFeatureTable(nFrames, nFeatures);

  tc->sequentialMode = TRUE;
  tc->writeInternalImages = FALSE;
  tc->affineConsistencyCheck = -1;  /* set this to 2 to turn on affine consistency check */

  // Read the first image
  sprintf(fnamein, "newset/frame_%06d.pgm", startFrame);  // <-- changed %06d to %04d
  img1 = pgmReadFile(fnamein, NULL, &ncols, &nrows);
  img2 = (unsigned char *) malloc(ncols * nrows * sizeof(unsigned char));

  // ========== START TOTAL TIMING ==========
  gettimeofday(&tv_start, NULL);
  // ========================================

  // Select good features from the first image
  KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);
  // KLTStoreFeatureList(fl, ft, 0);  // Disabled for performance
  // sprintf(fnameout, "feat_%04d.ppm", startFrame);
  // KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, fnameout);  // Disabled for performance

  // Track features through all subsequent frames
  for (i = 1, frame = startFrame + 1; frame <= endFrame; i++, frame++) {
    sprintf(fnamein, "newset/frame_%06d.pgm", frame);  // <-- changed %06d to %04d
    pgmReadFile(fnamein, img2, &ncols, &nrows);

    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

#ifdef REPLACE
    KLTReplaceLostFeatures(tc, img2, ncols, nrows, fl);
#endif

    // KLTStoreFeatureList(fl, ft, i);  // Disabled for performance
    // sprintf(fnameout, "feat_%04d.ppm", frame);
    // KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, fnameout);  // Disabled for performance
  }

  // Save feature table results
  // KLTWriteFeatureTable(ft, "features.txt", "%5.1f");  // Disabled for performance
  // KLTWriteFeatureTable(ft, "features.ft", NULL);  // Disabled for performance

  // ========== STOP TOTAL TIMING ==========
  gettimeofday(&tv_stop, NULL);
  total_time_ms = (tv_stop.tv_sec - tv_start.tv_sec) * 1000.0 +
                  (tv_stop.tv_usec - tv_start.tv_usec) / 1000.0;
  // =======================================

  printf("\n");
  printf("╔══════════════════════════════════════════════════════════════════════╗\n");
  printf("║                    TOTAL EXECUTION TIME (V3-2 CPU)                   ║\n");
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

  return 0;
}

