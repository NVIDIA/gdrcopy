#!/bin/sh

scriptdir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"

tmpfolder=$(mktemp --tmpdir -d gdrcopy.XXXXXXXXX)

testfile="${tmpfolder}/test-dummy.c"
makefile="${tmpfolder}/Makefile"

cat >${testfile} <<EOF
#include <linux/module.h>
#include <linux/mm.h>
static int __init test_dummy_init(void)
{
    struct vm_area_struct vma;
    vm_flags_set(&vma, 0);
    return 0;
}

static void __exit test_dummy_fini(void)
{
}

MODULE_AUTHOR("gpudirect@nvidia.com");
MODULE_LICENSE("MIT");
MODULE_VERSION("1.0");

module_init(test_dummy_init);
module_exit(test_dummy_fini);
EOF

cp "${scriptdir}/makefile-template" ${makefile}

cd ${tmpfolder}
make > /dev/null 2>&1 
ret=$?

cd ${scriptdir}
rm -rf ${tmpfolder}

if [ "${ret}" -eq 0 ]; then
    echo "y"
else
    echo "n"
fi

