/*********************************************************************
 * gpu_memory_pool.h
 * 
 * GPU Memory Pool Manager - Eliminates repeated malloc/free overhead
 * Pre-allocates and reuses GPU buffers across frames
 *********************************************************************/

#ifndef _GPU_MEMORY_POOL_H_
#define _GPU_MEMORY_POOL_H_

#include <cuda_runtime.h>

// Maximum image dimensions (adjust based on your use case)
#define MAX_IMAGE_WIDTH  4096
#define MAX_IMAGE_HEIGHT 4096
#define MAX_WINDOW_SIZE  128

// GPU Memory Pool Structure
typedef struct {
    // Image buffers (keep on GPU between frames)
    float *d_img1;           // Current frame
    float *d_img2;           // Next frame  
    float *d_gradx1;         // Gradients for frame 1
    float *d_grady1;
    float *d_gradx2;         // Gradients for frame 2
    float *d_grady2;
    
    // Pyramid buffers for multiple levels (images and gradients)
    // Structure: d_pyramid[level][0]=img, d_pyramid[level][1]=gradx, d_pyramid[level][2]=grady
    float ***d_pyramid1;     // Pyramid for frame 1 [nLevels][3]
    float ***d_pyramid2;     // Pyramid for frame 2 [nLevels][3]
    
    // Window buffers (small, reusable)
    float *d_imgdiff;        // Intensity difference window
    float *d_gradx_out;      // Gradient output windows
    float *d_grady_out;
    
    // Pinned (page-locked) host memory for faster transfers
    float *h_pinned_img1;
    float *h_pinned_img2;
    float *h_pinned_gradx1;
    float *h_pinned_grady1;
    float *h_pinned_gradx2;
    float *h_pinned_grady2;
    
    // Buffer sizes (track for validation)
    int img_size_bytes;
    int max_window_size_bytes;
    int current_ncols, current_nrows;
    int n_pyramid_levels;
    int *pyramid_ncols;      // Per-level dimensions
    int *pyramid_nrows;
    
    // Pool initialized flag
    int initialized;
    
    // CUDA stream for async operations
    cudaStream_t stream;
    
} GPU_MemoryPool;

// Initialize GPU memory pool (call once at startup)
extern "C" int GPU_InitMemoryPool(int max_ncols, int max_nrows, int max_window_size);

// Initialize with pyramid support (call once at startup)
extern "C" int GPU_InitMemoryPoolPyramid(int max_ncols, int max_nrows, int max_window_size, int n_levels);

// Cleanup GPU memory pool (call at shutdown)
extern "C" void GPU_FreeMemoryPool();

// Get/set image buffers (for keeping images on GPU)
extern "C" float* GPU_GetImageBuffer(int buffer_id);
extern "C" void GPU_UploadImage(float *h_img, int ncols, int nrows, int buffer_id);
extern "C" void GPU_DownloadImage(float *h_img, int ncols, int nrows, int buffer_id);

// Pyramid buffer management (upload once, reuse many times)
extern "C" void GPU_UploadPyramidLevel(int pyramid_id, int level, float *h_img, float *h_gradx, float *h_grady, int ncols, int nrows);
extern "C" float* GPU_GetPyramidImage(int pyramid_id, int level);
extern "C" float* GPU_GetPyramidGradX(int pyramid_id, int level);
extern "C" float* GPU_GetPyramidGradY(int pyramid_id, int level);

// Get pinned host memory (for faster transfers)
extern "C" float* GPU_GetPinnedHostBuffer(int buffer_id);

// Get window buffers (for small temporary windows)
extern "C" float* GPU_GetWindowBuffer(int buffer_id, int width, int height);

// Synchronize GPU operations
extern "C" void GPU_Sync();

// Set pyramid level dimensions (for proper indexing)
extern "C" void GPU_SetPyramidDims(int level, int ncols, int nrows);

#endif // _GPU_MEMORY_POOL_H_

