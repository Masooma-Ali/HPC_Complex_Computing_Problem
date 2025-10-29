#!/bin/bash
# Script to generate GPU profile on remote server

echo "=== GPU Profile Generation Script ==="
echo ""
echo "Step 1: Generating profile with nsys..."
nsys profile -o klt_gpu_profile --stats=true ./example3_gpu

echo ""
echo "Step 2: Extracting statistics..."
nsys stats --report gputrace klt_gpu_profile.nsys-rep > klt_gpu_stats.txt
nsys stats --report cudaapisum klt_gpu_profile.nsys-rep >> klt_gpu_stats.txt

echo ""
echo "Profile generation complete!"
echo "Files created:"
echo "  - klt_gpu_profile.nsys-rep (binary profile)"
echo "  - klt_gpu_stats.txt (text statistics)"
echo ""
echo "Transfer to local machine with:"
echo "  scp 23I-0743@172.17.170.89:/home/23I-0743/klt/klt_gpu_stats.txt ."


