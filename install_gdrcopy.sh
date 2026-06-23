#!/bin/bash

#git clone -b master https://github.com/NVIDIA/gdrcopy.git /tmp/gdrcopy
cd /tmp/gdrcopy/
mkdir prefix
make PREFIX=./prefix/ CUDA=/usr/local/cuda/ all install

echo "### Run insmod.sh ###"
apt-get update && apt-get install -y kmod
./insmod_root.sh

echo "### lsmod ###"
lsmod |grep "gdr"

echo "### validate ###"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-9.2/lib64/:$PWD/prefix/lib64
./validate

echo "### copybw ###"
./copybw
