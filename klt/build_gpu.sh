#!/bin/bash
# Build script for GPU-accelerated KLT

set -e  # Exit on error

echo "=========================================="
echo "KLT GPU Build Script"
echo "=========================================="

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: nvcc (CUDA compiler) not found!"
    echo "Please install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "✓ Found nvcc: $(nvcc --version | grep release)"

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "WARNING: nvidia-smi not found. GPU may not be available."
else
    echo "✓ GPU detected:"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
fi

# Fix filename typo if needed
if [ -f "convole.cu" ]; then
    echo "→ Fixing filename: convole.cu → convolve.cu"
    mv convole.cu convolve.cu
fi

# Clean previous build
echo "→ Cleaning previous build..."
rm -f *.o *.a example3_gpu

# Compile CPU source files
echo "→ Compiling CPU source files..."
gcc -c -DNDEBUG -O3 error.c pnmio.c pyramid.c selectGoodFeatures.c \
    storeFeatures.c klt.c klt_util.c writeFeatures.c

# Compile GPU source files
echo "→ Compiling GPU source files (convolve.cu and trackfeatures.cu)..."
nvcc -c -arch=sm_35 -O3 convolve.cu trackfeatures.cu

# Create library
echo "→ Creating library..."
ar ruv libklt_gpu.a *.o

# Compile example3
echo "→ Compiling example3..."
nvcc -arch=sm_35 -O3 -o example3_gpu example3.c -L. -lklt_gpu -lm

# Cleanup object files
rm -f *.o

echo ""
echo "=========================================="
echo "✓ Build successful!"
echo "=========================================="
echo ""
echo "To run the example:"
echo "  ./example3_gpu"
echo ""
echo "Note: This uses FULL GPU acceleration:"
echo "      - GPU convolution (convolve.cu)"
echo "      - GPU feature tracking (trackfeatures.cu)"
echo ""

