#!/bin/bash
######################################################################
# OpenACC Build Script for V4 KLT Tracker
# Phase 1 Implementation
######################################################################

echo "======================================================"
echo " Building V4 OpenACC KLT Tracker (Phase 1)"
echo "======================================================"

# Clean previous build
echo "Cleaning previous build..."
make clean

# Build with OpenACC
echo ""
echo "Building with OpenACC support..."
make all

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================"
    echo " Build SUCCESSFUL!"
    echo "======================================================"
    echo ""
    echo "Executable: example3"
    echo ""
    echo "To run:"
    echo "  ./example3"
    echo ""
    echo "OpenACC Features Enabled:"
    echo "  ✓ GPU-accelerated convolution"
    echo "  ✓ GPU-accelerated feature selection (eigenvalue computation)"
    echo "  ✓ Persistent GPU data regions"
    echo "  ✓ Optimized I/O (reduced PPM writes)"
    echo ""
else
    echo ""
    echo "======================================================"
    echo " Build FAILED!"
    echo "======================================================"
    echo ""
    echo "Please check compiler errors above."
    echo ""
    echo "Note: Ensure pgcc or nvc compiler is available:"
    echo "  which pgcc"
    echo "  which nvc"
    echo ""
    exit 1
fi

