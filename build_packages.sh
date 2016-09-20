set -x

make clean

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
cp -r * $tmpdir/gdrcopy/
rm -f $tmpdir/gdrcopy-$VERSION.tar.gz
cd $tmpdir
mv gdrcopy gdrcopy-$VERSION
tar czvf gdrcopy-$VERSION.tar.gz gdrcopy-$VERSION

mkdir -p $tmpdir/topdir/{SRPMS,RPMS,SPECS,BUILD,SOURCES}
cp gdrcopy-$VERSION/gdrcopy.spec $tmpdir/topdir/SPECS/
cp gdrcopy-$VERSION.tar.gz $tmpdir/topdir/SOURCES/

rpmbuild -ba --nodeps --define "_topdir $tmpdir/topdir" --define 'dist %{nil}' --define 'CUDA /usr/local/cuda-7.5' --define "KVERSION $(uname -r)" $tmpdir/topdir/SPECS/gdrcopy.spec
rpms=`ls -1 $tmpdir/topdir/RPMS/*/*.rpm`
srpm=`ls -1 $tmpdir/topdir/SRPMS/`
echo $srpm $rpms
cd $cwd
mv $tmpdir/topdir/SRPMS/*.rpm .
mv $tmpdir/topdir/RPMS/*/*.rpm .

rm -rf $tmpdir
