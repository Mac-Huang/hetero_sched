# ============================================================================
# Heterogeneous Task Scheduler Makefile
# ============================================================================

# Compiler settings
CC = gcc
NVCC = nvcc

# Directories
SRCDIR = src
INCDIR = include
TASKDIR = tasks
BUILDDIR = build
OBJDIR = $(BUILDDIR)/obj

# Target executable
TARGET = $(BUILDDIR)/hetero_sched

# Compiler flags
CFLAGS = -Wall -Wextra -O3 -std=c99 -I$(INCDIR)
CFLAGS += -pthread -lm

# CUDA flags
CUDA_FLAGS = -O3 -I$(INCDIR) -arch=sm_35 -lcublas
CUDA_LIBS = -lcudart -lcublas

# Debug flags (use make DEBUG=1)
ifdef DEBUG
    CFLAGS += -g -DDEBUG -O0
    CUDA_FLAGS += -g -G
endif

# CUDA support (use make CUDA=1)
ifdef CUDA
    CFLAGS += -DUSE_CUDA
    CUDA_ENABLED = 1
endif

# Source files
C_SOURCES = $(wildcard $(SRCDIR)/*.c)
TASK_SOURCES = $(wildcard $(TASKDIR)/*.c)
CUDA_SOURCES = $(wildcard $(SRCDIR)/*.cu)

# Filter out ml_predict.c if no ML support wanted
ifndef ML
    C_SOURCES := $(filter-out $(SRCDIR)/ml_predict.c,$(C_SOURCES))
endif

# Object files
C_OBJECTS = $(C_SOURCES:$(SRCDIR)/%.c=$(OBJDIR)/%.o)
TASK_OBJECTS = $(TASK_SOURCES:$(TASKDIR)/%.c=$(OBJDIR)/%.o)
CUDA_OBJECTS = $(CUDA_SOURCES:$(SRCDIR)/%.cu=$(OBJDIR)/%.cu.o)

# All objects
ifdef CUDA_ENABLED
    ALL_OBJECTS = $(C_OBJECTS) $(TASK_OBJECTS) $(CUDA_OBJECTS)
    LDFLAGS += $(CUDA_LIBS)
else
    ALL_OBJECTS = $(C_OBJECTS) $(TASK_OBJECTS)
endif

# Default target
.PHONY: all
all: $(TARGET)

# Create build directories
$(OBJDIR):
	mkdir -p $(OBJDIR)

# Main target
$(TARGET): $(OBJDIR) $(ALL_OBJECTS)
	@echo "Linking $(TARGET)..."
ifdef CUDA_ENABLED
	$(NVCC) $(ALL_OBJECTS) -o $(TARGET) $(LDFLAGS)
else
	$(CC) $(ALL_OBJECTS) -o $(TARGET) $(CFLAGS)
endif
	@echo "Build complete!"

# Compile C source files
$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Compile task source files
$(OBJDIR)/%.o: $(TASKDIR)/%.c
	@echo "Compiling $<..."
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files
ifdef CUDA_ENABLED
$(OBJDIR)/%.cu.o: $(SRCDIR)/%.cu
	@echo "Compiling CUDA $<..."
	$(NVCC) $(CUDA_FLAGS) -c $< -o $@
endif

# ============================================================================
# Build Variants
# ============================================================================

# CPU-only build
.PHONY: cpu
cpu:
	$(MAKE) clean
	$(MAKE) all

# CUDA build
.PHONY: cuda
cuda:
	$(MAKE) clean
	$(MAKE) all CUDA=1

# Debug build
.PHONY: debug
debug:
	$(MAKE) clean
	$(MAKE) all DEBUG=1

# Debug CUDA build  
.PHONY: debug-cuda
debug-cuda:
	$(MAKE) clean
	$(MAKE) all DEBUG=1 CUDA=1

# Release build with optimizations
.PHONY: release
release: CFLAGS += -DNDEBUG -march=native -flto
release: CUDA_FLAGS += -DNDEBUG
release:
	$(MAKE) clean
	$(MAKE) all

# ============================================================================
# Testing and Benchmarks
# ============================================================================

# Run the program
.PHONY: run
run: $(TARGET)
	./$(TARGET)

# Run with custom parameters
.PHONY: run-test
run-test: $(TARGET)
	./$(TARGET) --tasks 20

# Run benchmark suite
.PHONY: benchmark
benchmark: $(TARGET)
	./$(TARGET) --benchmark

# Run CPU benchmarks only
.PHONY: benchmark-cpu
benchmark-cpu:
	$(MAKE) cpu
	./$(TARGET) --benchmark

# Run CUDA benchmarks
.PHONY: benchmark-cuda
benchmark-cuda:
	$(MAKE) cuda
	./$(TARGET) --benchmark

# Performance comparison
.PHONY: compare
compare:
	@echo "=== CPU-only Performance ==="
	$(MAKE) benchmark-cpu
	@echo ""
	@echo "=== CUDA Performance ==="
	$(MAKE) benchmark-cuda

# ============================================================================
# Individual Task Tests
# ============================================================================

# Test vector addition tasks
.PHONY: test-vector
test-vector: $(TARGET)
	./$(TARGET) --tasks 10 | grep "VEC_ADD"

# Test matrix multiplication tasks  
.PHONY: test-matrix
test-matrix: $(TARGET)
	./$(TARGET) --tasks 10 | grep "MATMUL"

# ============================================================================
# Development Tools
# ============================================================================

# Static analysis with cppcheck
.PHONY: analyze
analyze:
	cppcheck --enable=all --inconclusive --std=c99 $(SRCDIR)/ $(INCDIR)/ $(TASKDIR)/

# Format code with clang-format
.PHONY: format
format:
	find $(SRCDIR) $(INCDIR) $(TASKDIR) -name "*.c" -o -name "*.h" -o -name "*.cu" | \
	xargs clang-format -i -style="{IndentWidth: 4, UseTab: Never}"

# Generate documentation
.PHONY: docs
docs:
	doxygen Doxyfile

# ============================================================================
# Installation and Packaging
# ============================================================================

PREFIX ?= /usr/local
BINDIR = $(PREFIX)/bin

.PHONY: install
install: $(TARGET)
	mkdir -p $(BINDIR)
	cp $(TARGET) $(BINDIR)/
	chmod +x $(BINDIR)/hetero_sched

.PHONY: uninstall
uninstall:
	rm -f $(BINDIR)/hetero_sched

# Create distribution package
.PHONY: dist
dist: clean
	tar -czf hetero_sched.tar.gz \
		$(SRCDIR)/ $(INCDIR)/ $(TASKDIR)/ \
		Makefile README.md

# ============================================================================
# Cleanup
# ============================================================================

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)
	rm -f *.o *.cu.o

.PHONY: distclean
distclean: clean
	rm -f hetero_sched.tar.gz
	rm -rf docs/

# ============================================================================
# Help
# ============================================================================

.PHONY: help
help:
	@echo "Heterogeneous Task Scheduler Build System"
	@echo ""
	@echo "Basic Targets:"
	@echo "  all          - Build with current configuration"
	@echo "  cpu          - Build CPU-only version"
	@echo "  cuda         - Build with CUDA support"
	@echo "  debug        - Build debug version"
	@echo "  debug-cuda   - Build debug version with CUDA"
	@echo "  release      - Build optimized release version"
	@echo ""
	@echo "Testing:"
	@echo "  run          - Run with default parameters"
	@echo "  run-test     - Run with test parameters"
	@echo "  benchmark    - Run benchmark suite"
	@echo "  benchmark-cpu - Run CPU-only benchmarks"
	@echo "  benchmark-cuda - Run CUDA benchmarks"
	@echo "  compare      - Compare CPU vs CUDA performance"
	@echo ""
	@echo "Task Tests:"
	@echo "  test-vector  - Test vector addition tasks"
	@echo "  test-matrix  - Test matrix multiplication tasks"
	@echo ""
	@echo "Development:"
	@echo "  analyze      - Run static analysis"
	@echo "  format       - Format source code"
	@echo "  docs         - Generate documentation"
	@echo ""
	@echo "Installation:"
	@echo "  install      - Install to system (PREFIX=$(PREFIX))"
	@echo "  uninstall    - Remove from system"
	@echo "  dist         - Create distribution package"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean        - Remove build files"
	@echo "  distclean    - Remove all generated files"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1      - Enable debug build"
	@echo "  CUDA=1       - Enable CUDA support"
	@echo "  PREFIX=path  - Set installation prefix"

# ============================================================================
# Dependencies
# ============================================================================

# Automatic dependency generation
-include $(ALL_OBJECTS:.o=.d)

# Generate dependencies for C files
$(OBJDIR)/%.d: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -MM -MT '$(OBJDIR)/$*.o' $< > $@

$(OBJDIR)/%.d: $(TASKDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -MM -MT '$(OBJDIR)/$*.o' $< > $@

.PHONY: deps
deps: $(C_OBJECTS:.o=.d) $(TASK_OBJECTS:.o=.d)