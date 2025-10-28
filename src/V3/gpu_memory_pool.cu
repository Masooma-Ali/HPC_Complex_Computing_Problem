/*********************************************************************
 * gpu_memory_pool.cu
 * 
 * GPU Memory Pool Implementation - Persistent GPU memory management
 *********************************************************************/

#include "gpu_memory_pool.h"
#include <stdlib.h>
#include <stdio.h>

// KLT_verbose declaration (defined in klt.c)
extern "C" int KLT_verbose;

#define cudaCheckError(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPU Memory Pool error: %s %s %d\n", 
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Global memory pool
static GPU_MemoryPool g_pool = {0};

/*********************************************************************
 * Initialize GPU memory pool
 * Allocates all buffers upfront to avoid repeated allocations
 *********************************************************************/
extern "C" int GPU_InitMemoryPool(int max_ncols, int max_nrows, int max_window_size)
{
    // Calculate sizes
    int img_size = max_ncols * max_nrows * sizeof(float);
    int window_size = max_window_size * max_window_size * sizeof(float);
    
    // Allocate large image buffers
    cudaCheckError(cudaMalloc(&g_pool.d_img1, img_size));
    cudaCheckError(cudaMalloc(&g_pool.d_img2, img_size));
    cudaCheckError(cudaMalloc(&g_pool.d_gradx1, img_size));
    cudaCheckError(cudaMalloc(&g_pool.d_grady1, img_size));
    cudaCheckError(cudaMalloc(&g_pool.d_gradx2, img_size));
    cudaCheckError(cudaMalloc(&g_pool.d_grady2, img_size));
    
    // Allocate smaller window buffers (reused frequently)
    cudaCheckError(cudaMalloc(&g_pool.d_imgdiff, window_size));
    cudaCheckError(cudaMalloc(&g_pool.d_gradx_out, window_size));
    cudaCheckError(cudaMalloc(&g_pool.d_grady_out, window_size));
    
    // Create CUDA stream for async operations
    cudaCheckError(cudaStreamCreate(&g_pool.stream));
    
    // Store sizes
    g_pool.img_size_bytes = img_size;
    g_pool.max_window_size_bytes = window_size;
    g_pool.current_ncols = max_ncols;
    g_pool.current_nrows = max_nrows;
    g_pool.initialized = 1;
    
    if (KLT_verbose >= 1) {
        fprintf(stderr, "(GPU Memory Pool) Initialized: %dx%d images, %dx%d windows\n",
                max_ncols, max_nrows, max_window_size, max_window_size);
    }
    
    return 0;
}

/*********************************************************************
 * Initialize GPU memory pool with pyramid support
 * Allocates buffers for multiple pyramid levels
 *********************************************************************/
extern "C" int GPU_InitMemoryPoolPyramid(int max_ncols, int max_nrows, int max_window_size, int n_levels)
{
    if (GPU_InitMemoryPool(max_ncols, max_nrows, max_window_size) != 0) {
        return -1;
    }
    
    g_pool.n_pyramid_levels = n_levels;
    
    // Allocate pyramid dimension arrays
    g_pool.pyramid_ncols = (int*)malloc(n_levels * sizeof(int));
    g_pool.pyramid_nrows = (int*)malloc(n_levels * sizeof(int));
    
    // Allocate pyramid buffers for frame 1 and 2
    g_pool.d_pyramid1 = (float***)malloc(n_levels * sizeof(float**));
    g_pool.d_pyramid2 = (float***)malloc(n_levels * sizeof(float**));
    
    // Calculate sizes for each pyramid level and allocate
    for (int level = 0; level < n_levels; level++) {
        int level_cols = max_ncols / (1 << level);
        int level_rows = max_nrows / (1 << level);
        int level_size = level_cols * level_rows * sizeof(float);
        
        g_pool.pyramid_ncols[level] = level_cols;
        g_pool.pyramid_nrows[level] = level_rows;
        
        // Allocate 3 buffers per level: [img, gradx, grady]
        g_pool.d_pyramid1[level] = (float**)malloc(3 * sizeof(float*));
        g_pool.d_pyramid2[level] = (float**)malloc(3 * sizeof(float*));
        
        for (int i = 0; i < 3; i++) {
            cudaCheckError(cudaMalloc(&g_pool.d_pyramid1[level][i], level_size));
            cudaCheckError(cudaMalloc(&g_pool.d_pyramid2[level][i], level_size));
        }
    }
    
    // Allocate pinned host memory for faster transfers
    int img_size = max_ncols * max_nrows * sizeof(float);
    cudaCheckError(cudaHostAlloc(&g_pool.h_pinned_img1, img_size, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&g_pool.h_pinned_img2, img_size, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&g_pool.h_pinned_gradx1, img_size, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&g_pool.h_pinned_grady1, img_size, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&g_pool.h_pinned_gradx2, img_size, cudaHostAllocDefault));
    cudaCheckError(cudaHostAlloc(&g_pool.h_pinned_grady2, img_size, cudaHostAllocDefault));
    
    if (KLT_verbose >= 1) {
        fprintf(stderr, "(GPU Memory Pool) Pyramid initialized: %d levels\n", n_levels);
    }
    
    return 0;
}

/*********************************************************************
 * Free GPU memory pool
 *********************************************************************/
extern "C" void GPU_FreeMemoryPool()
{
    if (!g_pool.initialized) return;
    
    // Free standard buffers
    if (g_pool.d_img1) cudaFree(g_pool.d_img1);
    if (g_pool.d_img2) cudaFree(g_pool.d_img2);
    if (g_pool.d_gradx1) cudaFree(g_pool.d_gradx1);
    if (g_pool.d_grady1) cudaFree(g_pool.d_grady1);
    if (g_pool.d_gradx2) cudaFree(g_pool.d_gradx2);
    if (g_pool.d_grady2) cudaFree(g_pool.d_grady2);
    if (g_pool.d_imgdiff) cudaFree(g_pool.d_imgdiff);
    if (g_pool.d_gradx_out) cudaFree(g_pool.d_gradx_out);
    if (g_pool.d_grady_out) cudaFree(g_pool.d_grady_out);
    if (g_pool.stream) cudaStreamDestroy(g_pool.stream);
    
    // Free pyramid buffers
    if (g_pool.d_pyramid1 && g_pool.n_pyramid_levels > 0) {
        for (int level = 0; level < g_pool.n_pyramid_levels; level++) {
            if (g_pool.d_pyramid1[level]) {
                for (int i = 0; i < 3; i++) {
                    if (g_pool.d_pyramid1[level][i]) cudaFree(g_pool.d_pyramid1[level][i]);
                }
                free(g_pool.d_pyramid1[level]);
            }
        }
        free(g_pool.d_pyramid1);
    }
    
    if (g_pool.d_pyramid2 && g_pool.n_pyramid_levels > 0) {
        for (int level = 0; level < g_pool.n_pyramid_levels; level++) {
            if (g_pool.d_pyramid2[level]) {
                for (int i = 0; i < 3; i++) {
                    if (g_pool.d_pyramid2[level][i]) cudaFree(g_pool.d_pyramid2[level][i]);
                }
                free(g_pool.d_pyramid2[level]);
            }
        }
        free(g_pool.d_pyramid2);
    }
    
    // Free pinned host memory
    if (g_pool.h_pinned_img1) cudaFreeHost(g_pool.h_pinned_img1);
    if (g_pool.h_pinned_img2) cudaFreeHost(g_pool.h_pinned_img2);
    if (g_pool.h_pinned_gradx1) cudaFreeHost(g_pool.h_pinned_gradx1);
    if (g_pool.h_pinned_grady1) cudaFreeHost(g_pool.h_pinned_grady1);
    if (g_pool.h_pinned_gradx2) cudaFreeHost(g_pool.h_pinned_gradx2);
    if (g_pool.h_pinned_grady2) cudaFreeHost(g_pool.h_pinned_grady2);
    
    // Free pyramid dimension arrays
    if (g_pool.pyramid_ncols) free(g_pool.pyramid_ncols);
    if (g_pool.pyramid_nrows) free(g_pool.pyramid_nrows);
    
    g_pool.initialized = 0;
    
    if (KLT_verbose >= 1) {
        fprintf(stderr, "(GPU Memory Pool) Freed\n");
    }
}

/*********************************************************************
 * Get image buffer pointer
 * buffer_id: 0=img1, 1=img2, 2=gradx1, 3=grady1, 4=gradx2, 5=grady2
 *********************************************************************/
extern "C" float* GPU_GetImageBuffer(int buffer_id)
{
    if (!g_pool.initialized) {
        fprintf(stderr, "ERROR: GPU Memory Pool not initialized!\n");
        return NULL;
    }
    
    switch(buffer_id) {
        case 0: return g_pool.d_img1;
        case 1: return g_pool.d_img2;
        case 2: return g_pool.d_gradx1;
        case 3: return g_pool.d_grady1;
        case 4: return g_pool.d_gradx2;
        case 5: return g_pool.d_grady2;
        default: return NULL;
    }
}

/*********************************************************************
 * Upload image to GPU buffer (only when needed)
 * buffer_id: 0=img1, 1=img2, etc.
 *********************************************************************/
extern "C" void GPU_UploadImage(float *h_img, int ncols, int nrows, int buffer_id)
{
    if (!g_pool.initialized) return;
    
    int size = ncols * nrows * sizeof(float);
    float *d_img = GPU_GetImageBuffer(buffer_id);
    
    if (d_img) {
        cudaCheckError(cudaMemcpyAsync(d_img, h_img, size, 
                                       cudaMemcpyHostToDevice, g_pool.stream));
    }
}

/*********************************************************************
 * Download image from GPU buffer (only when needed)
 *********************************************************************/
extern "C" void GPU_DownloadImage(float *h_img, int ncols, int nrows, int buffer_id)
{
    if (!g_pool.initialized) return;
    
    int size = ncols * nrows * sizeof(float);
    float *d_img = GPU_GetImageBuffer(buffer_id);
    
    if (d_img) {
        cudaCheckError(cudaMemcpyAsync(h_img, d_img, size,
                                       cudaMemcpyDeviceToHost, g_pool.stream));
    }
}

/*********************************************************************
 * Get window buffer pointer
 * buffer_id: 0=imgdiff, 1=gradx_out, 2=grady_out
 *********************************************************************/
extern "C" float* GPU_GetWindowBuffer(int buffer_id, int width, int height)
{
    if (!g_pool.initialized) {
        fprintf(stderr, "ERROR: GPU Memory Pool not initialized!\n");
        return NULL;
    }
    
    int window_size = width * height * sizeof(float);
    if (window_size > g_pool.max_window_size_bytes) {
        fprintf(stderr, "ERROR: Window size %dx%d exceeds max %dx%d\n",
                width, height, (int)sqrt(g_pool.max_window_size_bytes/sizeof(float)),
                (int)sqrt(g_pool.max_window_size_bytes/sizeof(float)));
        return NULL;
    }
    
    switch(buffer_id) {
        case 0: return g_pool.d_imgdiff;
        case 1: return g_pool.d_gradx_out;
        case 2: return g_pool.d_grady_out;
        default: return NULL;
    }
}

/*********************************************************************
 * Synchronize GPU operations
 *********************************************************************/
extern "C" void GPU_Sync()
{
    if (g_pool.initialized) {
        cudaCheckError(cudaStreamSynchronize(g_pool.stream));
    }
}

/*********************************************************************
 * Upload pyramid level to GPU (images and gradients)
 * Upload once, reuse many times for all features at this level
 *********************************************************************/
extern "C" void GPU_UploadPyramidLevel(int pyramid_id, int level, float *h_img, float *h_gradx, float *h_grady, int ncols, int nrows)
{
    if (!g_pool.initialized || level >= g_pool.n_pyramid_levels) return;
    
    int size = ncols * nrows * sizeof(float);
    float ***pyramid = (pyramid_id == 0) ? g_pool.d_pyramid1 : g_pool.d_pyramid2;
    
    // Upload all three buffers (img, gradx, grady) asynchronously
    if (h_img && pyramid[level][0]) {
        cudaCheckError(cudaMemcpyAsync(pyramid[level][0], h_img, size, 
                                       cudaMemcpyHostToDevice, g_pool.stream));
    }
    if (h_gradx && pyramid[level][1]) {
        cudaCheckError(cudaMemcpyAsync(pyramid[level][1], h_gradx, size,
                                       cudaMemcpyHostToDevice, g_pool.stream));
    }
    if (h_grady && pyramid[level][2]) {
        cudaCheckError(cudaMemcpyAsync(pyramid[level][2], h_grady, size,
                                       cudaMemcpyHostToDevice, g_pool.stream));
    }
}

/*********************************************************************
 * Get pyramid image/gradient pointers (already on GPU)
 *********************************************************************/
extern "C" float* GPU_GetPyramidImage(int pyramid_id, int level)
{
    if (!g_pool.initialized || level >= g_pool.n_pyramid_levels) return NULL;
    float ***pyramid = (pyramid_id == 0) ? g_pool.d_pyramid1 : g_pool.d_pyramid2;
    return pyramid[level][0];
}

extern "C" float* GPU_GetPyramidGradX(int pyramid_id, int level)
{
    if (!g_pool.initialized || level >= g_pool.n_pyramid_levels) return NULL;
    float ***pyramid = (pyramid_id == 0) ? g_pool.d_pyramid1 : g_pool.d_pyramid2;
    return pyramid[level][1];
}

extern "C" float* GPU_GetPyramidGradY(int pyramid_id, int level)
{
    if (!g_pool.initialized || level >= g_pool.n_pyramid_levels) return NULL;
    float ***pyramid = (pyramid_id == 0) ? g_pool.d_pyramid1 : g_pool.d_pyramid2;
    return pyramid[level][2];
}

/*********************************************************************
 * Get pinned host memory buffer (for faster transfers)
 *********************************************************************/
extern "C" float* GPU_GetPinnedHostBuffer(int buffer_id)
{
    if (!g_pool.initialized) return NULL;
    
    switch(buffer_id) {
        case 0: return g_pool.h_pinned_img1;
        case 1: return g_pool.h_pinned_img2;
        case 2: return g_pool.h_pinned_gradx1;
        case 3: return g_pool.h_pinned_grady1;
        case 4: return g_pool.h_pinned_gradx2;
        case 5: return g_pool.h_pinned_grady2;
        default: return NULL;
    }
}

/*********************************************************************
 * Set pyramid level dimensions
 *********************************************************************/
extern "C" void GPU_SetPyramidDims(int level, int ncols, int nrows)
{
    if (level < g_pool.n_pyramid_levels && g_pool.pyramid_ncols && g_pool.pyramid_nrows) {
        g_pool.pyramid_ncols[level] = ncols;
        g_pool.pyramid_nrows[level] = nrows;
    }
}

