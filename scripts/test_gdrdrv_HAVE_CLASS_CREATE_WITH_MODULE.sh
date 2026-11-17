#!/bin/sh

show_help()
{
    echo "Usage: ${0} [-hk]"
    echo
    echo "  -h          Show this help text."
    echo "  -k <kver>   Specify the kernel version."
    echo
}

set_kver=0
kver=""

OPTIND=1    # Reset in case getopts has been used previously in the shell.
while getopts "hk:" opt ; do
    case "${opt}" in
        h)
            show_help
            exit 0
            ;;
        k)
            set_kver=1
            kver="${OPTARG}"
            ;;
        ?)
            show_help
            exit 1
            ;;
    esac
done

if [ ${set_kver} -eq 0 ]; then
    kver="$(uname -r)"
fi

kdir="/lib/modules/${kver}/build"

tmpfolder=$(mktemp --tmpdir -d gdrcopy.XXXXXXXXX) || exit 1

testfile="${tmpfolder}/test-dummy.c"
makefile="${tmpfolder}/Makefile"

cat >"${testfile}" <<EOF
#include <linux/module.h>
#include <linux/device.h>
#include <linux/err.h>
static int __init test_dummy_init(void)
{
    struct class *cls;
    cls = class_create(THIS_MODULE, "test-dummy");
    if (IS_ERR(cls))
        return PTR_ERR(cls);
    class_destroy(cls);
    return 0;
}

static void __exit test_dummy_fini(void)
{
}

MODULE_AUTHOR("gpudirect@nvidia.com");
MODULE_LICENSE("Dual MIT/GPL");
MODULE_VERSION("1.0");

module_init(test_dummy_init);
module_exit(test_dummy_fini);
EOF

cat >"${makefile}" <<EOF
obj-m := test-dummy.o
EOF

cd "${tmpfolder}" || {
    rm -rf "${tmpfolder}"
    exit 1
}
make -C "${kdir}" M="${tmpfolder}" modules > /dev/null 2>&1
ret=$?

rm -rf "${tmpfolder}"

if [ "${ret}" -eq 0 ]; then
    echo "y"
else
    echo "n"
fi
