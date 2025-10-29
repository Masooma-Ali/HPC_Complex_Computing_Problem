# GPU-accelerated KLT Makefile with Enhanced Profiling
CC = gcc
NVCC = nvcc
ARCH = sm_75

# Profiling tool configuration
PROFILER ?= auto
NSYS_STATS_ARGS = --force-export=true,csv
NVPROF_METRICS = --metrics gst_throughput,gld_throughput,dram_throughput,warp_execution_efficiency,branch_efficiency,inst_executed_per_warp,issue_throughput

.PHONY: all clean run profile profile-basic profile-detailed profile-nsys profile-nvprof profile-metrics profile-report

all: example3_gpu

example3_gpu: libklt_gpu.a
	$(NVCC) -arch=$(ARCH) -O3 -o example3_gpu example3.c -L. -lklt_gpu -lm

libklt_gpu.a: cpu_objs gpu_objs
	ar ruv libklt_gpu.a *.o

cpu_objs:
	$(CC) -c -DNDEBUG -O3 error.c pnmio.c pyramid.c storeFeatures.c klt.c klt_util.c writeFeatures.c

gpu_objs:
	$(NVCC) -c -arch=$(ARCH) -O3 convole.cu trackfeatures.cu selectGoodFeatures.cu

run: example3_gpu
	./example3_gpu

# ============================================================================
# PROFILING TARGETS
# ============================================================================

# Auto-detect and run best available profiler
profile: example3_gpu
	@if command -v nsys >/dev/null 2>&1; then \
		echo "=== GPU Profiling with NVIDIA Nsys (Latest) ==="; \
		$(MAKE) profile-nsys; \
	elif command -v nvprof >/dev/null 2>&1; then \
		echo "=== GPU Profiling with NVIDIA nvprof ==="; \
		$(MAKE) profile-nvprof; \
	else \
		echo "=== Performance Measurement (time only) ==="; \
		time ./example3_gpu; \
		echo ""; \
		echo "For GPU profiling, install:"; \
		echo "  nsys: apt-get install nsight-systems-cli"; \
		echo "  nvprof: Usually comes with CUDA toolkit"; \
	fi

# Basic profiling (lightweight, fast)
profile-basic: example3_gpu
	@echo "=== Running Basic GPU Profile ===" ;\
	if command -v nsys >/dev/null 2>&1; then \
		nsys profile --trace=cuda,cudnn --output klt_gpu_profile_basic ./example3_gpu; \
		echo "Profile saved to klt_gpu_profile_basic.nsys-rep"; \
	else \
		nvprof --print-gpu-trace ./example3_gpu; \
	fi

# Detailed profiling with metrics and statistics
profile-detailed: example3_gpu
	@echo "=== Running Detailed GPU Profile with Metrics ===" ;\
	if command -v nsys >/dev/null 2>&1; then \
		nsys profile --trace=cuda,cudnn,osrt --output klt_gpu_profile_detailed --stats=true ./example3_gpu; \
		echo "Profile saved to klt_gpu_profile_detailed.nsys-rep"; \
		echo "Generating statistics report..."; \
		nsys stats klt_gpu_profile_detailed.nsys-rep --report "cuda_api_sum,cuda_kernel_sum,gpu_mem_time_sum" > klt_gpu_stats_detailed.txt 2>&1; \
		echo "Statistics saved to klt_gpu_stats_detailed.txt"; \
	else \
		nvprof --print-gpu-trace --log-file nvprof_detailed.txt ./example3_gpu; \
		echo "Profile saved to nvprof_detailed.txt"; \
	fi

# NSYS-specific profiling (System-wide profiling)
profile-nsys: example3_gpu
	@echo "=== NVIDIA Nsys Profiling ===" ;\
	@echo "1. Trace GPU and CUDA API calls..."; \
	nsys profile --trace=cuda,cudnn --output klt_gpu_profile_nsys ./example3_gpu; \
	@echo ""; \
	@echo "2. Generating statistical reports..."; \
	@echo "   - CUDA API Summary"; \
	nsys stats klt_gpu_profile_nsys.nsys-rep --report cuda_api_sum -f csv > klt_nsys_cuda_api.csv 2>&1; \
	@echo "   - CUDA Kernel Summary"; \
	nsys stats klt_gpu_profile_nsys.nsys-rep --report cuda_kernel_sum -f csv > klt_nsys_kernel_sum.csv 2>&1; \
	@echo "   - GPU Memory Activity"; \
	nsys stats klt_gpu_profile_nsys.nsys-rep --report gpu_mem_time_sum -f csv > klt_nsys_gpu_mem.csv 2>&1; \
	@echo ""; \
	@echo "Profile and reports saved:"; \
	@echo "  Main report: klt_gpu_profile_nsys.nsys-rep"; \
	@echo "  CSV reports: klt_nsys_*.csv"; \
	@ls -lh klt_gpu_profile_nsys.nsys-rep klt_nsys_*.csv 2>/dev/null

# NVPROF-specific profiling (Detailed kernel metrics)
profile-nvprof: example3_gpu
	@echo "=== NVIDIA nvprof Profiling ===" ;\
	@echo "1. Running trace with metrics..."; \
	nvprof --print-gpu-trace --log-file nvprof_trace.txt ./example3_gpu; \
	@echo ""; \
	@echo "2. Running detailed metrics analysis..."; \
	nvprof $(NVPROF_METRICS) --log-file nvprof_metrics.txt ./example3_gpu; \
	@echo ""; \
	@echo "Reports saved:"; \
	@echo "  Trace: nvprof_trace.txt"; \
	@echo "  Metrics: nvprof_metrics.txt"; \
	@ls -lh nvprof_*.txt 2>/dev/null

# Memory profiling
profile-memory: example3_gpu
	@echo "=== GPU Memory Profiling ===" ;\
	if command -v nsys >/dev/null 2>&1; then \
		echo "Using nsys for memory analysis..."; \
		nsys profile --trace=cuda,cudnn --output klt_gpu_profile_memory ./example3_gpu; \
		nsys stats klt_gpu_profile_memory.nsys-rep --report gpu_mem_time_sum -f csv > klt_nsys_memory_report.csv; \
		echo "Memory profile saved to klt_nsys_memory_report.csv"; \
	else \
		nvprof --print-gpu-trace --metrics gld_throughput,gst_throughput,dram_throughput ./example3_gpu | tee nvprof_memory.txt; \
	fi

# Comprehensive report generation
profile-report: example3_gpu
	@echo "=== Comprehensive GPU Profiling Report ===" ;\
	@echo "Timestamp: $$(date)" > klt_profile_report.txt; \
	@echo "======================================" >> klt_profile_report.txt; \
	@echo "" >> klt_profile_report.txt; \
	@echo "System Information:" >> klt_profile_report.txt; \
	@nvidia-smi >> klt_profile_report.txt 2>&1; \
	@echo "" >> klt_profile_report.txt; \
	@echo "Running GPU Profile..." ;\
	if command -v nsys >/dev/null 2>&1; then \
		echo "Using nsys profiler..." >> klt_profile_report.txt; \
		nsys profile --trace=cuda,cudnn --output klt_gpu_profile_comprehensive --stats=true ./example3_gpu >> klt_profile_report.txt 2>&1; \
		echo "" >> klt_profile_report.txt; \
		echo "=== CUDA API Statistics ===" >> klt_profile_report.txt; \
		nsys stats klt_gpu_profile_comprehensive.nsys-rep --report cuda_api_sum >> klt_profile_report.txt 2>&1; \
		echo "" >> klt_profile_report.txt; \
		echo "=== CUDA Kernel Statistics ===" >> klt_profile_report.txt; \
		nsys stats klt_gpu_profile_comprehensive.nsys-rep --report cuda_kernel_sum >> klt_profile_report.txt 2>&1; \
		echo "" >> klt_profile_report.txt; \
		echo "=== GPU Memory Statistics ===" >> klt_profile_report.txt; \
		nsys stats klt_gpu_profile_comprehensive.nsys-rep --report gpu_mem_time_sum >> klt_profile_report.txt 2>&1; \
	else \
		echo "Using nvprof profiler..." >> klt_profile_report.txt; \
		nvprof --print-gpu-trace --metrics gst_throughput,gld_throughput,dram_throughput,warp_execution_efficiency,branch_efficiency ./example3_gpu >> klt_profile_report.txt 2>&1; \
	fi; \
	echo "" >> klt_profile_report.txt; \
	echo "=== Report Generation Complete ===" >> klt_profile_report.txt; \
	echo "Report saved to: klt_profile_report.txt"; \
	@echo ""; \
	@cat klt_profile_report.txt

# Run with custom metrics
profile-custom: example3_gpu
	@echo "=== Custom Profiling with User Metrics ===" ;\
	@if command -v nsys >/dev/null 2>&1; then \
		echo "Enter metrics (e.g., cuda_api_sum,cuda_kernel_sum):"; \
		read metrics; \
		nsys stats klt_gpu_profile_nsys.nsys-rep --report $$metrics > klt_nsys_custom.txt 2>&1; \
		echo "Custom report saved to klt_nsys_custom.txt"; \
	else \
		echo "nvprof custom metrics mode"; \
		nvprof --print-gpu-trace $(NVPROF_METRICS) ./example3_gpu; \
	fi

# Benchmark comparison (run multiple times and compare)
profile-benchmark: example3_gpu
	@echo "=== GPU Benchmark Profile ===" ;\
	@echo "Running 3 iterations for performance analysis..."; \
	@for i in 1 2 3; do \
		echo "Iteration $$i..."; \
		if command -v nsys >/dev/null 2>&1; then \
			nsys profile --trace=cuda --output klt_gpu_benchmark_$$i ./example3_gpu > /dev/null 2>&1; \
		else \
			nvprof --print-gpu-trace ./example3_gpu > nvprof_benchmark_$$i.txt 2>&1; \
		fi; \
	done; \
	@echo "Benchmark complete"

clean:
	rm -f *.o libklt_gpu.a example3_gpu klt_gpu_profile* klt_nsys_* nvprof_* \
	      klt_gpu_stats* klt_profile_report.txt *.sqlite *.csv *.txt 2>/dev/null; \
	@echo "Cleaned build artifacts and profiling data"

help:
	@echo "=== GPU KLT Makefile Targets ===" ;\
	@echo ""; \
	@echo "Build targets:"; \
	@echo "  make                    - Build GPU executable"; \
	@echo "  make clean              - Remove build artifacts"; \
	@echo ""; \
	@echo "Run targets:"; \
	@echo "  make run                - Run GPU example"; \
	@echo ""; \
	@echo "Profiling targets:"; \
	@echo "  make profile            - Auto-detect and run best profiler"; \
	@echo "  make profile-basic      - Fast lightweight profiling"; \
	@echo "  make profile-detailed   - Detailed profiling with metrics"; \
	@echo "  make profile-nsys       - Use nsys profiler (recommended)"; \
	@echo "  make profile-nvprof     - Use nvprof profiler"; \
	@echo "  make profile-memory     - Memory-focused profiling"; \
	@echo "  make profile-report     - Generate comprehensive report"; \
	@echo "  make profile-benchmark  - Run 3 iterations for comparison"; \
	@echo ""; \
	@echo "Examples:"; \
	@echo "  make profile-nsys       - Full nsys profiling with stats"; \
	@echo "  make profile-detailed   - Detailed metrics and statistics"; \
	@echo "  make profile-report     - Complete analysis report"; \
