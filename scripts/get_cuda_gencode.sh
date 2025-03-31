#!/bin/bash

# Get CUDA version from nvcc
NVCC="${1:-nvcc}"
CUDA_VERSION_FULL=$("$NVCC" --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
CUDA_VERSION_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f1)
CUDA_VERSION_MINOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f2)

if [ -z "$CUDA_VERSION_FULL" ]; then
    echo "Error: Could not detect CUDA version" >&2
    exit 1
fi

# Require CUDA >= 8.0
if [ "$CUDA_VERSION_MAJOR" -lt 8 ]; then
    echo "Error: CUDA version must be >= 8.0" >&2
    exit 1
fi

# Get the list of supported SM architectures (sm_XX) from nvcc
# Filter to only include SM >= 60 (Pascal)
ARCH_LIST=$("$NVCC" --list-gpu-code 2>/dev/null | sed 's/sm_//' | awk '$1 >= 60')

if [ -z "$ARCH_LIST" ]; then
    echo "Warning: Could not determine supported architectures from nvcc, falling back to version-based detection" >&2

    # Initialize with Pascal architecture
    ARCH_LIST="60 61 62"

    # Add Volta (7.0) if CUDA >= 9.0
    if [ "$CUDA_VERSION_MAJOR" -ge 9 ]; then
        ARCH_LIST="$ARCH_LIST 70 72"
    fi

    # Add Turing (7.5) if CUDA >= 10.0
    if [ "$CUDA_VERSION_MAJOR" -ge 10 ]; then
        ARCH_LIST="$ARCH_LIST 75"
    fi

    # Add Ampere (8.0, 8.6, 8.7) if CUDA >= 11.1
    if [ "$CUDA_VERSION_MAJOR" -ge 11 ] && [ "$CUDA_VERSION_MINOR" -ge 1 ]; then
        ARCH_LIST="$ARCH_LIST 80 86 87"
    fi

    # Add Ada Lovelace (8.9) if CUDA >= 11.8
    if [ "$CUDA_VERSION_MAJOR" -ge 11 ] && [ "$CUDA_VERSION_MINOR" -ge 8 ]; then
        ARCH_LIST="$ARCH_LIST 89"
    fi

    # Add Hopper (9.0) if CUDA >= 12.0
    if [ "$CUDA_VERSION_MAJOR" -ge 12 ]; then
        ARCH_LIST="$ARCH_LIST 90"
    fi

    # Add Blackwell (10.0, 10.1, 12.0) if CUDA >= 12.8
    if [ "$CUDA_VERSION_MAJOR" -ge 12 ] && [ "$CUDA_VERSION_MINOR" -ge 8 ]; then
        ARCH_LIST="$ARCH_LIST 100 101 120"
    fi
fi

# Generate NVCC flags
GENCODE_FLAGS=""

# Generate SM-specific code for all architectures
for arch in $ARCH_LIST; do
    GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$arch,code=sm_$arch"
done

# Generate PTX code only for the latest architecture
LATEST_ARCH=$(echo "$ARCH_LIST" | tr ' ' '\n' | sort -n | tail -1)
GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$LATEST_ARCH,code=compute_$LATEST_ARCH"

echo "$GENCODE_FLAGS"
