#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
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


# Restart this number at 1 if MAJOR_VERSION or MINOR_VERSION changes
# See https://rpm-packaging-guide.github.io/#preamble-items
RPM_VERSION=1

SCRIPT_DIR_PATH="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TOP_DIR_PATH="${SCRIPT_DIR_PATH}/.."

CWD=$(pwd)

ex()
{
    local rc
    echo "+ $@"
    $@
    rc=$?
    
    if [[ $rc -ne 0 ]]; then
        echo "Failed with error $rc to execute: $@" >&2
        exit $rc
    fi
}


function show_help
{
    echo "This script is for generating GDRCopy RPM packages."
    echo
    echo "Usage: CUDA=<path> $0 [-m]"
    echo
    echo "Optional arguments:"
    echo "  -m              Generate non-dkms kmod package (default: no)."
    echo
    echo "Environment variables:"
    echo "  CUDA=<path>             [Required] CUDA installation path (usually /usr/local/cuda)."
    echo "  NVIDIA_SRC_DIR=<path>   [Optional] NVIDIA driver source directory (usually /usr/src/nvidia-<version>/nvidia)."
}

OPTIND=1	# Reset in case getopts has been used previously in the shell.

generate_kmod_nondkms=0

while getopts "h?m" opt; do
    case "$opt" in
    h|\?)
        show_help
        exit 0
        ;;
    m)  generate_kmod_nondkms=1
        ;;
    esac
done

shift $((OPTIND-1))


if [ "X$CUDA" == "X" ]; then
    echo "CUDA environment variable is not defined"
    exit 1
fi

echo "Building rpm package ..."

ex cd ${SCRIPT_DIR_PATH}

MODULE_SUBDIR=$(awk '/MODULE_SUBDIR \?=/ { print $3 }' ${TOP_DIR_PATH}/src/gdrdrv/Makefile | tr -d '\n')

MAJOR_VERSION=$(awk '/#define GDR_API_MAJOR_VERSION/ { print $3 }' ${TOP_DIR_PATH}/include/gdrapi.h | tr -d '\n')
MINOR_VERSION=$(awk '/#define GDR_API_MINOR_VERSION/ { print $3 }' ${TOP_DIR_PATH}/include/gdrapi.h | tr -d '\n')
VERSION="${MAJOR_VERSION}.${MINOR_VERSION}"
if [ "X$VERSION" == "X" ]; then
    echo "Failed to get version numbers!" >&2
    exit 1
fi
FULL_VERSION="${VERSION}"

if [[ ${generate_kmod_nondkms} == 1 ]]; then
    if [ -z "${NVIDIA_SRC_DIR}" ]; then
        NVIDIA_SRC_DIR=$(find /usr/src/nvidia-* -name "nv-p2p.h" -print -quit)
        if [ ${#NVIDIA_SRC_DIR} -gt 0 ]; then
            NVIDIA_SRC_DIR=$(dirname ${NVIDIA_SRC_DIR})
        fi
    fi

    if [ -d ${NVIDIA_SRC_DIR} ]; then
        NVIDIA_DRIVER_VERSION=$(basename $(dirname ${NVIDIA_SRC_DIR}))
    else
        echo "NVIDIA_SRC_DIR=${NVIDIA_SRC_DIR}" >&2
        echo "Failed to find NVIDIA driver!" >&2
        exit 1
    fi
fi


tmpdir=`mktemp -d /tmp/gdr.XXXXXX`
if [ ! -d "$tmpdir" ]; then
    echo "Failed to create a temp directory!" >&2
    exit 1
fi

echo "Building gdrcopy rpm packages version ${VERSION} ..."

echo "Working in $tmpdir ..."

ex cd ${TOP_DIR_PATH}

ex mkdir -p $tmpdir/gdrcopy
ex rm -rf $tmpdir/gdrcopy/*
ex cp -r packages/dkms.conf packages/rhel/init.d insmod.sh Makefile README.md include src tests config_arch LICENSE packages/gdrcopy.spec $tmpdir/gdrcopy/
ex rm -f $tmpdir/gdrcopy-$VERSION.tar.gz

ex cd $tmpdir/gdrcopy
ex find . -type f -exec sed -i "s/@FULL_VERSION@/${FULL_VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@VERSION@/${VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@MODULE_LOCATION@/${MODULE_SUBDIR//\//\\/}/g" {} +

ex cd $tmpdir
ex mv gdrcopy gdrcopy-$VERSION
ex tar czvf gdrcopy-$VERSION.tar.gz gdrcopy-$VERSION

ex mkdir -p $tmpdir/topdir/{SRPMS,RPMS,SPECS,BUILD,SOURCES}
ex cp gdrcopy-$VERSION/gdrcopy.spec $tmpdir/topdir/SPECS/
ex cp gdrcopy-$VERSION.tar.gz $tmpdir/topdir/SOURCES/

rpmbuild_params="-ba --nodeps --define '_build_id_links none' --define \"_topdir $tmpdir/topdir\" --define \"_release ${RPM_VERSION}\" --define 'dist %{nil}' --define \"CUDA $CUDA\" --define \"GDR_VERSION ${VERSION}\" --define \"KVERSION $(uname -r)\" --define \"MODULE_LOCATION ${MODULE_SUBDIR}\""
if [[ ${generate_kmod_nondkms} == 1 ]]; then
    rpmbuild_params="${rpmbuild_params} --define \"NVIDIA_DRIVER_VERSION ${NVIDIA_DRIVER_VERSION}\" --define \"NVIDIA_SRC_DIR ${NVIDIA_SRC_DIR}\" --define \"BUILD_KMOD_NONDKMS 1\""
fi
rpmbuild_params="${rpmbuild_params} $tmpdir/topdir/SPECS/gdrcopy.spec"
eval "rpmbuild ${rpmbuild_params}"

#rpmbuild -ba --nodeps --define '_build_id_links none' --define "_topdir $tmpdir/topdir" --define "_release ${RPM_VERSION}" --define 'dist %{nil}' --define "CUDA $CUDA" --define "GDR_VERSION ${VERSION}" --define "KVERSION $(uname -r)" --define "MODULE_LOCATION ${MODULE_SUBDIR}" --define "NVIDIA_DRIVER_VERSION ${NVIDIA_DRIVER_VERSION}" --define "NVIDIA_SRC_DIR ${NVIDIA_SRC_DIR}" $tmpdir/topdir/SPECS/gdrcopy.spec

rpms=`ls -1 $tmpdir/topdir/RPMS/*/*.rpm`
srpm=`ls -1 $tmpdir/topdir/SRPMS/`
echo $srpm $rpms
ex cd ${CWD}
ex cp $tmpdir/topdir/SRPMS/*.rpm .
ex cp $tmpdir/topdir/RPMS/*/*.rpm .

echo
echo "Cleaning up ..."

ex rm -rf ${tmpdir}
