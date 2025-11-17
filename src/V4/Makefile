######################################################################
# OpenACC compiler configuration for V4
CC = pgcc
# Alternative: CC = nvc

######################################################################
# -DNDEBUG prevents the assert() statements from being included in 
# the code.  If you are having problems running the code, you might 
# want to comment this line to see if an assert() statement fires.
FLAG1 = -DNDEBUG

######################################################################
# -DKLT_USE_QSORT forces the code to use the standard qsort() 
# routine.  Otherwise it will use a quicksort routine that takes
# advantage of our specific data structure to greatly reduce the
# running time on some machines.  Uncomment this line if for some
# reason you are unhappy with the special routine.
# FLAG2 = -DKLT_USE_QSORT

######################################################################
# OpenACC compilation flags (newer pgcc/nvc syntax)
ACCFLAGS = -acc -gpu=cc75 -Minfo=accel
OPTFLAGS = -fast -O3

######################################################################
# Combined flags
CFLAGS = $(FLAG1) $(FLAG2) $(ACCFLAGS) $(OPTFLAGS)


######################################################################
# There should be no need to modify anything below this line (but
# feel free to if you want).

EXAMPLES = example1.c example2.c example3.c example4.c example5.c
ARCH = convolve.c error.c pnmio.c pyramid.c selectGoodFeatures.c \
       storeFeatures.c trackFeatures.c klt.c klt_util.c writeFeatures.c
LIB = -L/usr/local/lib -L/usr/lib

.SUFFIXES:  .c .o
.PHONY: all lib clean depend profile profile1 profile2 profile3 profile4 profile5 build run

# Default target: just build
all: build

# Build everything
build: lib $(EXAMPLES:.c=)

# Compile object files
.c.o:
	$(CC) -c $(CFLAGS) $<

# Build library from object files
lib: $(ARCH:.c=.o)
	rm -f libklt.a
	ar ruv libklt.a $(ARCH:.c=.o)
	rm -f *.o

example1: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example2: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example3: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example4: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

example5: libklt.a
	$(CC) -O3 $(CFLAGS) -o $@ $@.c -L. -lklt $(LIB) -lm

# Run: build, execute, and profile
run: build
	@echo "========================================"
	@echo "Running example3..."
	@echo "========================================"
	./example3
	@echo "========================================"
	@echo "Execution completed!"
	@echo "========================================"
	@echo ""
	@echo "========================================"
	
# Profiling with nsys
profile: example3
	@echo "========================================"
	@echo "Profiling example3 with nsys..."
	@echo "========================================"
	nsys profile -o v4_profile --stats=true ./example3
	@echo "========================================"
	@echo "Profile data saved to v4_profile.nsys-rep"
	@echo "Run 'nsys stats v4_profile.nsys-rep' to view statistics"
	@echo "========================================"

profile1: example1
	nsys profile -o v4_profile_ex1 --stats=true ./example1

profile2: example2
	nsys profile -o v4_profile_ex2 --stats=true ./example2

profile3: example3
	nsys profile -o v4_profile_ex3 --stats=true ./example3

profile4: example4
	nsys profile -o v4_profile_ex4 --stats=true ./example4

profile5: example5
	nsys profile -o v4_profile_ex5 --stats=true ./example5

depend:
	makedepend $(ARCH) $(EXAMPLES)

clean:
	rm -f *.o *.a $(EXAMPLES:.c=) *.tar *.tar.gz libklt.a \
	      feat*.ppm features.ft features.txt *.nsys-rep *.sqlite



