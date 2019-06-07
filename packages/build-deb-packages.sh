#!/bin/bash

SCRIPT_DIR_PATH="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
TOP_DIR_PATH="${SCRIPT_DIR_PATH}/.."

CWD=$(pwd)

ex()
{
    if ! eval "$@"; then
        echo "Failed to execute: $@" >&2
        exit 1
    fi
}

set -x

if [ "X$CUDA" == "X" ]; then
    echo "CUDA is not defined"; exit 1
fi

cd ${SCRIPT_DIR_PATH}

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

cd ${TOP_DIR_PATH}

mkdir -p $tmpdir/gdrcopy
rm -rf $tmpdir/gdrcopy/*
cp -r autogen.sh configure.ac init.d insmod.sh Makefile.am README.md include src tests LICENSE packages/debian $tmpdir/gdrcopy/
rm -f $tmpdir/gdrcopy_$VERSION.orig.tar.gz

cd $tmpdir
mv gdrcopy gdrcopy-$VERSION
tar czvf gdrcopy_$VERSION.orig.tar.gz gdrcopy-$VERSION

cd $tmpdir/gdrcopy-$VERSION
debuild --set-envvar=CUDA=$CUDA -us -uc

cd ${CWD}
mv $tmpdir/*.deb .

