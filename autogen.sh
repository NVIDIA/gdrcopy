#/bin/sh

(set -x; libtoolize)
if [ $? -ne 0 ]
then
    echo "Error in libtoolize!!"
    exit 1
fi

(set -x; aclocal)
if [ $? -ne 0 ]
then
    echo "Error in aclocal!!"
    exit 1
fi

(set -x; autoconf)
if [ $? -ne 0 ]
then
    echo "Error in autoconf!!"
    exit 1
fi

(set -x; autoheader)
if [ $? -ne 0 ]
then
    echo "Error in autoheader!!"
    exit 1
fi

(set -x; automake --add-missing --foreign)
if [ $? -ne 0 ]
then
    echo "Error in automake!!"
    exit 1
fi
