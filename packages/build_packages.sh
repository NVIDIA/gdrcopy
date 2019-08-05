#!/bin/bash
set -x

ex()
{
    if ! eval "$@"; then
        echo "Failed to execute: $@" >&2
        exit 1
    fi
}

make clean

if [ "X$CUDA" == "X" ]; then
    echo "CUDA is not defined"; exit 1
fi

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

cwd=$PWD

#cp gdrcopy.spec ~/work/mellanox/rpmbuild/SPECS/

mkdir -p $tmpdir/gdrcopy
rm -rf $tmpdir/gdrcopy/*
cp -r *.* gdrdrv LICENSE config_arch Makefile $tmpdir/gdrcopy/
rm -f $tmpdir/gdrcopy-$VERSION.tar.gz
cd $tmpdir
mv gdrcopy gdrcopy-$VERSION
tar czvf gdrcopy-$VERSION.tar.gz gdrcopy-$VERSION

mkdir -p $tmpdir/topdir/{SRPMS,RPMS,SPECS,BUILD,SOURCES}
cp gdrcopy-$VERSION/gdrcopy.spec $tmpdir/topdir/SPECS/
cp gdrcopy-$VERSION.tar.gz $tmpdir/topdir/SOURCES/

rpmbuild -ba --nodeps --define "_topdir $tmpdir/topdir" --define 'dist %{nil}' --define "CUDA $CUDA"             --define "KVERSION $(uname -r)" $tmpdir/topdir/SPECS/gdrcopy.spec
rpms=`ls -1 $tmpdir/topdir/RPMS/*/*.rpm`
srpm=`ls -1 $tmpdir/topdir/SRPMS/`
echo $srpm $rpms
cd $cwd
mv $tmpdir/topdir/SRPMS/*.rpm .
mv $tmpdir/topdir/RPMS/*/*.rpm .

if false; then
echo "Building debian tarball for gdrcopy..."
# update version in changelog
sed -i -r "0,/^(.*) \(([a-zA-Z0-9.-]+)\) (.*)/s//\1 \(${VERSION}-${RELEASE}\) \3/" gdrcopy-${VERSION}/debian/changelog
ex tar czf gdrcopy_$VERSION.orig.tar.gz gdrcopy-$VERSION --exclude=.* --exclude=build_release.sh
ex mv gdrcopy_$VERSION.orig.tar.gz /tmp

/bin/rm -rf $tmpdir

echo ""
echo Built: /tmp/$srpm
echo Built: /tmp/nvidia-peer-memory_$VERSION.orig.tar.gz
echo ""
echo "To install run on RPM based OS:"
echo "    # rpmbuild --rebuild /tmp/$srpm"
echo "    # rpm -ivh <path to generated binary rpm file>" 
echo ""
echo "To install on DEB based OS:"
echo "    # cd /tmp"
echo "    # tar xzf /tmp/gdrcopy_$VERSION.orig.tar.gz"
echo "    # cd gdrcopy-$VERSION"
echo "    # dpkg-buildpackage -us -uc"
echo "    # dpkg -i <path to generated deb files>"
echo ""
fi
