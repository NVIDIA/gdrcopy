#/bin/sh

set -x

rm -rf aclocal.m4 autom4te.cache config config.h.in configure Makefile.in src/Makefile.in tests/Makefile.in

mkdir -p config/m4

autoreconf -iv || exit 1

