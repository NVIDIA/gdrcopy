#!/bin/bash

#nvidia-docker run -ti --privileged --cap-add=ALL -v $PWD/:/tmp/gdrcopy nvidia/cuda:11.0-devel-ubuntu16.04 bash
nvidia-docker run -ti --privileged --cap-add=ALL -v $PWD/:/tmp/gdrcopy nvidia/cuda:9.2-devel-ubuntu16.04 /tmp/gdrcopy/install_gdrcopy.sh
