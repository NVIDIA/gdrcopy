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

# Initialize with Pascal architecture
COMPUTE_LIST="60 61 62"
SM_LIST="60 61 62"

# Add Volta (7.0) if CUDA >= 9.0
if [ "$CUDA_VERSION_MAJOR" -ge 9 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 70 72"
    SM_LIST="$SM_LIST 70 72"
fi

# Add Turing (7.5) if CUDA >= 10.0
if [ "$CUDA_VERSION_MAJOR" -ge 10 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 75"
    SM_LIST="$SM_LIST 75"
fi

# Add Ampere (8.0, 8.6, 8.7) if CUDA >= 11.1
if [ "$CUDA_VERSION_MAJOR" -ge 11 ] && [ "$CUDA_VERSION_MINOR" -ge 1 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 80 86 87"
    SM_LIST="$SM_LIST 80 86 87"
fi

# Add Ada Lovelace (8.9) if CUDA >= 11.8
if [ "$CUDA_VERSION_MAJOR" -ge 11 ] && [ "$CUDA_VERSION_MINOR" -ge 8 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 89"
    SM_LIST="$SM_LIST 89"
fi

# Add Hopper (9.0) if CUDA >= 12.0
if [ "$CUDA_VERSION_MAJOR" -ge 12 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 90"
    SM_LIST="$SM_LIST 90"
fi

# Add Blackwell (10.0) if CUDA >= 12.6
if [ "$CUDA_VERSION_MAJOR" -ge 12 ] && [ "$CUDA_VERSION_MINOR" -ge 6 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 100"
    SM_LIST="$SM_LIST 100"
fi

# Add Blackwell (12.0) if CUDA >= 12.8
if [ "$CUDA_VERSION_MAJOR" -ge 12 ] && [ "$CUDA_VERSION_MINOR" -ge 8 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 120"
    SM_LIST="$SM_LIST 120"
fi

# Generate NVCC flags
GENCODE_FLAGS=""
for compute in $COMPUTE_LIST; do
    GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$compute,code=compute_$compute"
done

for sm in $SM_LIST; do
    GENCODE_FLAGS="$GENCODE_FLAGS -gencode arch=compute_$sm,code=sm_$sm"
done

echo "$GENCODE_FLAGS" 