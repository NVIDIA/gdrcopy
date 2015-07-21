#!/bin/bash

p2p_headers=$(find /usr/src/nvidia-* -name "nv-p2p.h")
p2p_header_count=$(echo "$p2p_headers" | wc -l)

if [ $p2p_header_count -eq 0 ]; then
    echo "ERROR: can't find sources of NVIDIA kernel-mode driver anywhere in /usr/src/nvidia-\*"
    exit 1
else
    dirname $(echo "$p2p_headers" | head -n 1)
    exit 0
fi
