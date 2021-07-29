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
# See https://www.debian.org/doc/debian-policy/ch-controlfields.html#version
DEBIAN_VERSION=1

SCRIPT_DIR_PATH="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TOP_DIR_PATH="${SCRIPT_DIR_PATH}/.."

CWD=$(pwd)

skip_dep_check=0
build_test_package=1
build_driver_package=1

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
    echo "Usage: [CUDA=<path>] $0 [-d] [-t] [-k] [-h]"
    echo ""
    echo "  CUDA=<path>     Set your installed CUDA path (ex. /usr/local/cuda)."
    echo "  -d              Don't check build dependencies. Use my environment variables such as C_INCLUDE_PATH instead."
    echo "  -t              Skip building gdrcopy-tests package."
    echo "  -k              Skip building gdrdrv-dkms package."
    echo "  -h              Show this help text."
    echo ""
}

OPTIND=1	# Reset in case getopts has been used previously in the shell.

while getopts "hdtk" opt; do
    case "${opt}" in
    h)
        show_help
        exit 0
        ;;
    d)  skip_dep_check=1
        ;;
    t)  build_test_package=0
        ;;
    k)  build_driver_package=0
        ;;
    esac
done

shift $((OPTIND-1))



if [[ ${build_test_package} == 1 ]] && [ "X$CUDA" == "X" ]; then
    echo "CUDA environment variable is not defined"; exit 1
fi

NVCC=${CUDA}/bin/nvcc
CUDA_VERSION=`$NVCC --version | grep release | sed 's/^.*release \([0-9]\+\.[0-9]\+\).*/\1/'`
CUDA_MAJOR=`echo ${CUDA_VERSION} | cut -d "." -f 1`
CUDA_MINOR=`echo ${CUDA_VERSION} | cut -d "." -f 2`

echo "Building debian package for the gdrcopy library ..."

ex cd ${SCRIPT_DIR_PATH}

MODULE_SUBDIR=$(awk '/MODULE_SUBDIR \?=/ { print $3 }' ${TOP_DIR_PATH}/src/gdrdrv/Makefile | tr -d '\n')

MAJOR_VERSION=$(awk '/#define GDR_API_MAJOR_VERSION/ { print $3 }' ${TOP_DIR_PATH}/include/gdrapi.h | tr -d '\n')
MINOR_VERSION=$(awk '/#define GDR_API_MINOR_VERSION/ { print $3 }' ${TOP_DIR_PATH}/include/gdrapi.h | tr -d '\n')
VERSION="${MAJOR_VERSION}.${MINOR_VERSION}"
if [ "X$VERSION" == "X" ]; then
    echo "Failed to get version numbers!" >&2
    exit 1
fi
FULL_VERSION="${VERSION}-${DEBIAN_VERSION}"

tmpdir=`mktemp -d /tmp/gdr.XXXXXX`
if [ ! -d "${tmpdir}" ]; then
    echo "Failed to create a temp directory!" >&2
    exit 1
fi

echo "Building gdrcopy debian packages version ${FULL_VERSION} ..."

echo "Working in ${tmpdir} ..."

ex cd ${TOP_DIR_PATH}

ex mkdir -p ${tmpdir}/gdrcopy
ex rm -rf ${tmpdir}/gdrcopy/*
ex cp -r Makefile README.md include src tests LICENSE config_arch ${tmpdir}/gdrcopy/
ex cp -r packages/debian-lib ${tmpdir}/gdrcopy/
ex cp -r packages/debian-tests ${tmpdir}/gdrcopy/
ex cp README.md ${tmpdir}/gdrcopy/debian-lib/README.Debian
ex cp README.md ${tmpdir}/gdrcopy/debian-lib/README.source
ex cp README.md ${tmpdir}/gdrcopy/debian-tests/README.Debian
ex cp README.md ${tmpdir}/gdrcopy/debian-tests/README.source

ex cd ${tmpdir}/gdrcopy
ex find . -type f -exec sed -i "s/@FULL_VERSION@/${FULL_VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@VERSION@/${VERSION}/g" {} +

ex rm -f ${tmpdir}/libgdrapi_${VERSION}.orig.tar.gz
ex rm -f ${tmpdir}/gdrcopy-tests_${VERSION}.orig.tar.gz

ex cd ${tmpdir}
ex cp -r gdrcopy libgdrapi-${VERSION}
ex cd ${tmpdir}/libgdrapi-${VERSION}
ex mv debian-lib debian
ex rm -rf debian-*

ex cd ${tmpdir}
ex cp -r gdrcopy gdrcopy-tests-${VERSION}
ex cd ${tmpdir}/gdrcopy-tests-${VERSION}
ex mv debian-tests debian
ex rm -rf debian-*

ex cd ${tmpdir}
ex tar czvf libgdrapi_${VERSION}.orig.tar.gz libgdrapi-${VERSION}
ex tar czvf gdrcopy-tests_${VERSION}.orig.tar.gz gdrcopy-tests-${VERSION}

echo "Building libgdrapi package ..."
ex cd ${tmpdir}/libgdrapi-${VERSION}
debuild_params="--set-envvar=PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"
if [ "${skip_dep_check}" -eq 1 ]; then
    debuild_params+=" --preserve-env -d"
    echo "Skip build dependency check. Use the environment variables instead ..."
fi
# --set-envvar needs to be placed before -us -uc
debuild_params+=" -us -uc"
ex debuild ${debuild_params}

if [[ ${build_test_package} == 1 ]]; then
    echo
    echo "Building gdrcopy-tests package ..."
    ex cd ${tmpdir}/gdrcopy-tests-${VERSION}
    debuild_params="--set-envvar=CUDA=${CUDA} --set-envvar=PKG_CONFIG_PATH=${PKG_CONFIG_PATH}"
    if [ "${skip_dep_check}" -eq 1 ]; then
        debuild_params+=" --preserve-env -d"
        echo "Skip build dependency check. Use the environment variables instead ..."
    fi
    # --set-envvar needs to be placed before -us -uc
    debuild_params+=" -us -uc"
    ex debuild ${debuild_params}
fi

if [[ ${build_driver_package} == 1 ]]; then
    echo
    echo "Building gdrdrv-dkms package ..."
    ex cd ${tmpdir}/gdrcopy/src/gdrdrv
    ex make clean

    dkmsdir="${tmpdir}/gdrdrv-dkms-${VERSION}"
    ex mkdir -p ${dkmsdir}
    ex cp -r ${tmpdir}/gdrcopy/src/gdrdrv ${dkmsdir}/gdrdrv-${VERSION}
    ex rm -rf ${dkmsdir}/gdrdrv-${VERSION}/debian-*
    ex cp ${SCRIPT_DIR_PATH}/dkms.conf ${dkmsdir}/gdrdrv-${VERSION}/
    ex cd ${dkmsdir}
    ex cp -r ${SCRIPT_DIR_PATH}/dkms/* .
    ex find . -type f -exec sed -i "s/@FULL_VERSION@/${FULL_VERSION}/g" {} +
    ex find . -type f -exec sed -i "s/@VERSION@/${VERSION}/g" {} +
    ex find . -type f -exec sed -i "s/@MODULE_LOCATION@/${MODULE_SUBDIR//\//\\/}/g" {} +

    ex cd ${tmpdir}
    ex tar czvf gdrdrv-dkms_${VERSION}.orig.tar.gz gdrdrv-dkms-${VERSION}

    ex cd ${dkmsdir}
    ex dpkg-buildpackage -rfakeroot -d -F -us -uc
fi

echo
echo "Building gdrcopy package ..."
metadir=${tmpdir}/gdrcopy-${VERSION}
ex mkdir -p ${metadir}
ex cd ${TOP_DIR_PATH}
ex cp -r packages/debian-meta ${metadir}/debian
ex cp README.md ${metadir}/debian/README.Debian
ex cp README.md ${metadir}/debian/README.source
ex cd ${metadir}
ex find . -type f -exec sed -i "s/@FULL_VERSION@/${FULL_VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@VERSION@/${VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@MODULE_LOCATION@/${MODULE_SUBDIR//\//\\/}/g" {} +
ex cd ${tmpdir}
ex tar czvf gdrcopy_${VERSION}.orig.tar.gz gdrcopy-${VERSION}
cd ${metadir}
ex debuild -us -uc

echo
echo "Copying *.deb and supplementary files to the current working directory ..."
if $(hash lsb_release 2>/dev/null); then
    release=`lsb_release -rs | sed -e "s/\./_/g"`
    id=`lsb_release -is | sed -e "s/ /_/g"`
    release=".${id}${release}"
else
    release=""
fi

ex cd ${CWD}

for item in `ls ${tmpdir}/*.deb`; do
    item_name=`basename $item`
    item_name=`echo $item_name | sed -e "s/\.deb//g"`
    if echo "$item_name" | grep -q "tests"; then
        item_name="${item_name}${release}+cuda${CUDA_MAJOR}.${CUDA_MINOR}.deb"
    else
        item_name="${item_name}${release}.deb"
    fi
    ex cp $item ./${item_name}
done
ex cp ${tmpdir}/*.tar.* .
ex cp ${tmpdir}/*.dsc .

echo
echo "Cleaning up ..."

ex rm -rf ${tmpdir}
