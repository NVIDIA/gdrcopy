#!/bin/bash

# Restart this number at 1 if MAJOR_VERSION or MINOR_VERSION changes
# See https://www.debian.org/doc/debian-policy/ch-controlfields.html#version
DEBIAN_VERSION=1

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

if [ "X$CUDA" == "X" ]; then
    echo "CUDA environment variable is not defined"; exit 1
fi

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
ex cp -r Makefile README.md include src tests LICENSE config_arch packages/debian ${tmpdir}/gdrcopy/
ex cp README.md ${tmpdir}/gdrcopy/debian/README.Debian
ex cp README.md ${tmpdir}/gdrcopy/debian/README.source
ex rm -f ${tmpdir}/gdrcopy_${VERSION}.orig.tar.gz

ex cd ${tmpdir}/gdrcopy
ex find . -type f -exec sed -i "s/@VERSION@/${FULL_VERSION}/g" {} +

ex cd ${tmpdir}
ex mv gdrcopy gdrcopy-${VERSION}
ex tar czvf gdrcopy_${VERSION}.orig.tar.gz gdrcopy-${VERSION}

ex cd ${tmpdir}/gdrcopy-${VERSION}
ex debuild --set-envvar=CUDA=${CUDA} --set-envvar=PKG_CONFIG_PATH=${PKG_CONFIG_PATH} -us -uc

echo
echo "Building dkms module ..."
ex cd ${tmpdir}/gdrcopy-${VERSION}/src/gdrdrv
ex make clean

ex mkdir -p ${tmpdir}/gdrdrv-dkms-${VERSION}/
ex cp -r ${tmpdir}/gdrcopy-${VERSION}/src/gdrdrv ${tmpdir}/gdrdrv-dkms-${VERSION}/gdrdrv-${VERSION}
ex cp ${SCRIPT_DIR_PATH}/dkms.conf ${tmpdir}/gdrdrv-dkms-${VERSION}/gdrdrv-${VERSION}/
ex cd ${tmpdir}/gdrdrv-dkms-${VERSION}/
ex cp -r ${SCRIPT_DIR_PATH}/dkms/* .
ex find . -type f -exec sed -i "s/@FULL_VERSION@/${FULL_VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@VERSION@/${VERSION}/g" {} +
ex find . -type f -exec sed -i "s/@MODULE_LOCATION@/${MODULE_SUBDIR//\//\\/}/g" {} +

ex dpkg-buildpackage -S -us -uc
ex dpkg-buildpackage -rfakeroot -d -b -us -uc

echo
echo "Copying *.deb and supplementary files to the current working directory ..."

ex cd ${CWD}
ex cp ${tmpdir}/*.deb .
ex cp ${tmpdir}/*.tar.* .
ex cp ${tmpdir}/*.dsc .

echo
echo "Cleaning up ..."

ex rm -rf ${tmpdir}
