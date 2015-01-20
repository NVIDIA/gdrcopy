#!/bin/bash

nv_path=""

for d in $(find /usr/src -name nvidia-\* -type d); do 
    if [ -e $d/nv-p2p.h ]; then 
        echo "$d"
        nv_path=$d
        break
    fi
done

if [ -z ${nv_path} ]; then
    echo "ERROR: can't find sources of NVIDIA kernel-mode driver anywhere in /usr/src/nvidia-\*"
    exit 1
fi

exit 0
