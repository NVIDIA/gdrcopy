#/bin/sh

set -x

libtoolize
aclocal
autoconf
automake --add-missing --foreign

