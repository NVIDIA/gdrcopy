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

if [ "X$CUDA" == "X" ]; then
    echo "CUDA environment variable is not defined"; exit 1
fi

echo "Building debian package for the gdrcopy library ..."

set -x

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

# Preparing for building the gdrdrv driver
cd $tmpdir/gdrcopy-$VERSION
./configure --with-cuda=$CUDA

set +x

echo
echo "Building dkms module ..."
echo "Request upgrading to root ..."

sudo -s <<EOF
set -x
mkdir -p /usr/src/gdrdrv-$VERSION/
cd /usr/src/gdrdrv-$VERSION/
cp -r $tmpdir/gdrcopy-$VERSION/src/gdrdrv/* .
cp ${SCRIPT_DIR_PATH}/dkms.conf .

dkms add -m gdrdrv -v $VERSION
dkms build -m gdrdrv -v $VERSION
dkms mkdsc -m gdrdrv -v $VERSION --source-only
dkms mkdeb -m gdrdrv -v $VERSION --source-only
cp /var/lib/dkms/gdrdrv/$VERSION/deb/*.deb $tmpdir/
dkms remove -m gdrdrv/$VERSION --all
rm -rf /var/lib/dkms/gdrdrv/$VERSION/
EOF

echo
echo "Copying *.deb to the current working directory ..."

set -x
cd ${CWD}
mv $tmpdir/*.deb .

