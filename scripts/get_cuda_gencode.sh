#!/bin/bash

# Get CUDA version from nvcc
NVCC="${1:-nvcc}"
CUDA_VERSION=$("$NVCC" --version | grep "release" | sed 's/.*release //' | sed 's/\..*//')

if [ -z "$CUDA_VERSION" ]; then
    echo "Error: Could not detect CUDA version" >&2
    exit 1
fi

# Initialize with Pascal architecture
COMPUTE_LIST="60 61 62"
SM_LIST="60 61 62"

# Add Volta (7.0) if CUDA >= 9.0
if [ "$CUDA_VERSION" -ge 9 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 70 72"
    SM_LIST="$SM_LIST 70 72"
fi

# Add Turing (7.5) if CUDA >= 10.0
if [ "$CUDA_VERSION" -ge 10 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 75"
    SM_LIST="$SM_LIST 75"
fi

# Add Ampere (8.0, 8.6) if CUDA >= 11.0
if [ "$CUDA_VERSION" -ge 11 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 80 86"
    SM_LIST="$SM_LIST 80 86"
fi

# Add Ada Lovelace (8.9) and Hopper (9.0) if CUDA >= 12.0
if [ "$CUDA_VERSION" -ge 12 ]; then
    COMPUTE_LIST="$COMPUTE_LIST 89 90"
    SM_LIST="$SM_LIST 89 90"
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