#!/bin/bash

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

tmpdir=`mktemp -d /tmp/gdr.XXXXXX`
if [ ! -d "$tmpdir" ]; then
    echo "Failed to create a temp directory!" >&2
    exit 1
fi

echo "Building gdrcopy rpm packages version ${VERSION} ..."

echo "Working in $tmpdir ..."

#cp gdrcopy.spec ~/work/mellanox/rpmbuild/SPECS/

ex cd ${TOP_DIR_PATH}

ex mkdir -p $tmpdir/gdrcopy
ex rm -rf $tmpdir/gdrcopy/*
ex cp -r packages/rhel/init.d insmod.sh Makefile README.md include src tests config_arch LICENSE packages/gdrcopy.spec $tmpdir/gdrcopy/
ex rm -f $tmpdir/gdrcopy-$VERSION.tar.gz

ex cd $tmpdir/gdrcopy
ex find . -type f -exec sed -i "s/@VERSION@/${VERSION}/g" {} +

ex cd $tmpdir
ex mv gdrcopy gdrcopy-$VERSION
ex tar czvf gdrcopy-$VERSION.tar.gz gdrcopy-$VERSION

ex mkdir -p $tmpdir/topdir/{SRPMS,RPMS,SPECS,BUILD,SOURCES}
ex cp gdrcopy-$VERSION/gdrcopy.spec $tmpdir/topdir/SPECS/
ex cp gdrcopy-$VERSION.tar.gz $tmpdir/topdir/SOURCES/

rpmbuild -ba --nodeps --define "_topdir $tmpdir/topdir" --define 'dist %{nil}' --define "CUDA $CUDA" --define "GDR_VERSION ${VERSION}" --define "KVERSION $(uname -r)" --define "MODULE_LOCATION ${MODULE_SUBDIR}" $tmpdir/topdir/SPECS/gdrcopy.spec
rpms=`ls -1 $tmpdir/topdir/RPMS/*/*.rpm`
srpm=`ls -1 $tmpdir/topdir/SRPMS/`
echo $srpm $rpms
ex cd ${CWD}
ex cp $tmpdir/topdir/SRPMS/*.rpm .
ex cp $tmpdir/topdir/RPMS/*/*.rpm .

