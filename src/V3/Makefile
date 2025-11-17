######################################################################
# GPU-accelerated KLT Makefile for V3-2
CC = gcc
NVCC = nvcc
ARCH = sm_75

######################################################################
# Compilation flags
CFLAGS = -DNDEBUG -O3
NVCCFLAGS = -arch=$(ARCH) -O3

######################################################################
# Source files
CPU_SRC = error.c pnmio.c pyramid.c storeFeatures.c klt.c klt_util.c writeFeatures.c
GPU_SRC = convole.cu trackfeatures.cu selectGoodFeatures.cu
EXAMPLES = example1.c example2.c example3.c example4.c example5.c

.SUFFIXES: .c .cu .o
.PHONY: all lib clean depend profile profile1 profile2 profile3 profile4 profile5 build run cpu_objs gpu_objs

# Default target: just build
all: build

# Build everything
build: lib example3_gpu

# Compile CPU object files
cpu_objs: $(CPU_SRC:.c=.o)

# Compile GPU object files
gpu_objs: $(GPU_SRC:.cu=.o)

# Compile .c files to .o
.c.o:
	$(CC) -c $(CFLAGS) $<

# Compile .cu files to .o
.cu.o:
	$(NVCC) -c $(NVCCFLAGS) $<

# Build library from object files
lib: cpu_objs gpu_objs
	rm -f libklt_gpu.a
	ar ruv libklt_gpu.a $(CPU_SRC:.c=.o) $(GPU_SRC:.cu=.o)
	rm -f $(CPU_SRC:.c=.o) $(GPU_SRC:.cu=.o)

example3_gpu: libklt_gpu.a
	$(NVCC) $(NVCCFLAGS) -o $@ example3.c -L. -lklt_gpu -lm

# Run: build and execute
run: build
	@echo "========================================"
	@echo "Running example3_gpu..."
	@echo "========================================"
	./example3_gpu
	@echo "========================================"
	@echo "Execution completed!"
	@echo "========================================"

# Profiling with nsys
profile: example3_gpu
	@echo "========================================"
	@echo "Profiling example3_gpu with nsys..."
	@echo "========================================"
	nsys profile --force-overwrite=true --trace=cuda --output v3-2_profile --stats=true ./example3_gpu
	@echo "========================================"
	@echo "Profile data saved to v3-2_profile.nsys-rep"
	@echo "Run 'nsys stats v3-2_profile.nsys-rep' to view statistics"
	@echo "========================================"

depend:
	@echo "Dependencies managed automatically"

clean:
	@echo "Cleaning build artifacts..."
	rm -f *.o *.a example3_gpu *.tar *.tar.gz libklt_gpu.a \
	      feat*.ppm features.ft features.txt *.nsys-rep *.sqlite \
	      v3-2_profile.* gpu_profile.* 2>/dev/null
	@echo "Clean completed!"
