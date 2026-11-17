#!/bin/bash
# Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

THIS_DIR=$(dirname "$0")

# remove driver
if grep -qw gdrdrv /proc/devices; then
    if ! sudo /sbin/rmmod gdrdrv; then
        echo "ERROR: could not unload the existing gdrdrv module" >&2
        exit 1
    fi
fi

# remove old inodes just in case
if [ -e /dev/gdrdrv ]; then
    sudo rm /dev/gdrdrv
fi

# insert driver
if ! sudo /sbin/insmod "$THIS_DIR/src/gdrdrv/gdrdrv.ko" dbg_enabled=0 info_enabled=0 use_persistent_mapping=1; then
    echo "ERROR: could not load gdrdrv" >&2
    exit 1
fi

# insmod bypasses modprobe.d; create/refresh the node the same way the package hook does.
if ! sudo "$THIS_DIR/scripts/gdrdrv_devnode.sh" create; then
    echo "ERROR: /dev/gdrdrv was not created" >&2
    exit 1
fi
if [ ! -e /dev/gdrdrv ]; then
    echo "ERROR: /dev/gdrdrv was not created" >&2
    exit 1
fi

echo "INFO: /dev/gdrdrv is ready"
ls -l /dev/gdrdrv
