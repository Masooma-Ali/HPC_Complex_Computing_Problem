/*
 * gpu_memory_pool.cu - GPU Memory Pool Implementation
 * Pre-allocates GPU memory once and reuses it for all operations
 * Eliminates repeated malloc/free overhead (63,720 calls → 1 call)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Global pool state
static struct {
    float *buffer_large;          // Image buffers (input, output, temp)
    float *buffer_large_2;        // Second large buffer for temp storage
    float *buffer_medium;         // Kernel buffer
    
    int large_size;               // Size in bytes
    int medium_size;              // Size in bytes
    
    int initialized;              // Flag to track initialization
} g_pool = {NULL, NULL, NULL, 0, 0, 0};

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (line %d)\n", cudaGetErrorString(err), __LINE__); \
            return -1; \
        } \
    } while(0)

/**
 * Initialize GPU memory pool
 * Allocates all buffers once at startup
 */
int gpu_memory_pool_init(int ncols, int nrows, int max_kernel_width)
{
    if (g_pool.initialized) {
        printf("GPU memory pool already initialized\n");
        return 0;
    }
    
    printf("[GPU Pool] Initializing memory pool...\n");
    printf("[GPU Pool] Image size: %d x %d = %d pixels\n", ncols, nrows, ncols * nrows);
    printf("[GPU Pool] Kernel width: %d\n", max_kernel_width);
    
    // Calculate buffer sizes
    g_pool.large_size = ncols * nrows * sizeof(float);
    g_pool.medium_size = max_kernel_width * sizeof(float);
    
    printf("[GPU Pool] Large buffer size: %.2f MB\n", g_pool.large_size / (1024.0f * 1024.0f));
    printf("[GPU Pool] Medium buffer size: %.2f 

 MB\n", g_pool.medium_size / (1024.0f * 1024.0f));
    
    // Allocate large buffer for image (input)
    CUDA_CHECK(cudaMalloc(&g_pool.buffer_large, g_pool.large_size));
    printf("[GPU Pool] ✓ Allocated large buffer 1 (%.2f MB)\n", g_pool.large_size / (1024.0f * 1024.0f));
    
    // Allocate large buffer for output/temp
    CUDA_CHECK(cudaMalloc(&g_pool.buffer_large_2, g_pool.large_size));
    printf("[GPU Pool] ✓ Allocated large buffer 2 (%.2f MB)\n", g_pool.large_size / (1024.0f * 1024.0f));
    
    // Allocate medium buffer for kernel
    CUDA_CHECK(cudaMalloc(&g_pool.buffer_medium, g_pool.medium_size));
    printf("[GPU Pool] ✓ Allocated medium buffer (%.2f KB)\n", g_pool.medium_size / 1024.0f);
    
    // Mark as initialized
    g_pool.initialized = 1;
    
    printf("[GPU Pool] Initialization complete!\n");
    printf("[GPU Pool] Total GPU memory allocated: %.2f MB\n", 
           (g_pool.large_size * 2 + g_pool.medium_size) / (1024.0f * 1024.0f));
    
    return 0;
}

/**
 * Get large buffer from pool
 * Returns pre-allocated device memory
 * No malloc overhead!
 */
float* gpu_get_large_buffer()
{
    if (!g_pool.initialized) {
        fprintf(stderr, "ERROR: GPU memory pool not initialized!\n");
        return NULL;
    }
    return g_pool.buffer_large;
}

/**
 * Get second large buffer (for temp/output)
 */
float* gpu_get_large_buffer_2()
{
    if (!g_pool.initialized) {
        fprintf(stderr, "ERROR: GPU memory pool not initialized!\n");
        return NULL;
    }
    return g_pool.buffer_large_2;
}

/**
 * Get medium buffer from pool (for kernels)
 */
float* gpu_get_medium_buffer()
{
    if (!g_pool.initialized) {
        fprintf(stderr, "ERROR: GPU memory pool not initialized!\n");
        return NULL;
    }
    return g_pool.buffer_medium;
}

/**
 * Reset pool for next use
 * (Currently no-op since memory stays allocated)
 */
void gpu_memory_pool_reset()
{
    // Memory is pre-allocated and stays allocated
    // This is a no-op but kept for API consistency
}

/**
 * Free all pool memory
 * Call once at program exit
 */
void gpu_memory_pool_free()
{
    if (!g_pool.initialized) {
        return;
    }
    
    printf("[GPU Pool] Freeing GPU memory pool...\n");
    
    if (g_pool.buffer_large) {
        cudaFree(g_pool.buffer_large);
        g_pool.buffer_large = NULL;
        printf("[GPU Pool] ✓ Freed large buffer 1\n");
    }
    
    if (g_pool.buffer_large_2) {
        cudaFree(g_pool.buffer_large_2);
        g_pool.buffer_large_2 = NULL;
        printf("[GPU Pool] ✓ Freed large buffer 2\n");
    }
    
    if (g_pool.buffer_medium) {
        cudaFree(g_pool.buffer_medium);
        g_pool.buffer_medium = NULL;
        printf("[GPU Pool] ✓ Freed medium buffer\n");
    }
    
    g_pool.initialized = 0;
    printf("[GPU Pool] Memory pool freed!\n");
}
