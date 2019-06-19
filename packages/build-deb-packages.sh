#!/bin/bash

SCRIPT_DIR_PATH="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TOP_DIR_PATH="${SCRIPT_DIR_PATH}/.."

CWD=$(pwd)

ex()
{
    local rc
    echo "+ $@"
    eval "$@"
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

VERSION=`grep Version: *.spec | cut -d : -f 2 | sed -e 's@\s@@g'`
RELEASE=`grep "define _release" *.spec | cut -d" " -f"4"| sed -r -e 's/}//'`
if [ "X$VERSION" == "X" ] || [ "X$RELEASE" == "X" ]; then
    echo "Failed to get version numbers!" >&2
    exit 1
fi

tmpdir=`mktemp -d /tmp/gdr.XXXXXX`
if [ ! -d "$tmpdir" ]; then
    echo "Failed to create a temp directory!" >&2
    exit 1
fi
echo "Working in $tmpdir ..."

ex cd ${TOP_DIR_PATH}

ex mkdir -p $tmpdir/gdrcopy
ex rm -rf $tmpdir/gdrcopy/*
ex cp -r autogen.sh configure.ac Makefile.am README.md include src tests LICENSE packages/debian $tmpdir/gdrcopy/
ex rm -f $tmpdir/gdrcopy_$VERSION.orig.tar.gz

ex cd $tmpdir
ex mv gdrcopy gdrcopy-$VERSION
ex tar czvf gdrcopy_$VERSION.orig.tar.gz gdrcopy-$VERSION

ex cd $tmpdir/gdrcopy-$VERSION
ex debuild --set-envvar=CUDA=$CUDA -us -uc

echo
echo "Building dkms module ..."
ex cd $tmpdir/gdrcopy-$VERSION
ex ./configure --with-cuda=$CUDA

ex mkdir -p $tmpdir/gdrdrv-dkms-$VERSION/
ex cp -r $tmpdir/gdrcopy-$VERSION/src/gdrdrv $tmpdir/gdrdrv-dkms-$VERSION/gdrdrv-$VERSION
ex cp ${SCRIPT_DIR_PATH}/dkms.conf $tmpdir/gdrdrv-dkms-$VERSION/gdrdrv-$VERSION/
ex cd $tmpdir/gdrdrv-dkms-$VERSION/
ex cp -r ${SCRIPT_DIR_PATH}/dkms/* .

ex dpkg-buildpackage -S -us -uc
ex dpkg-buildpackage -rfakeroot -d -b -us -uc

echo
echo "Copying *.deb and supplementary files to the current working directory ..."

ex cd ${CWD}
ex cp $tmpdir/*.deb .
ex cp $tmpdir/*.tar.* .
ex cp $tmpdir/*.dsc .

