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

THIS_DIR=$(dirname $0)

# remove driver
grep gdrdrv /proc/devices >/dev/null && sudo /sbin/rmmod gdrdrv

# insert driver
sudo /sbin/insmod src/gdrdrv/gdrdrv.ko dbg_enabled=0 info_enabled=0

# create device inodes
major=`fgrep gdrdrv /proc/devices | cut -b 1-4`
echo "INFO: driver major is $major"

# remove old inodes just in case
if [ -e /dev/gdrdrv ]; then
    sudo rm /dev/gdrdrv
fi

echo "INFO: creating /dev/gdrdrv inode"
sudo mknod /dev/gdrdrv c $major 0
sudo chmod a+w+r /dev/gdrdrv
