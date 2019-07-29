#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

# NOTE: the official suggestion for when to update the incremental version is:
# "If the library source code has changed at all since the last update, then
# increment revision (‘c:r:a’ becomes ‘c:r+1:a’)." from
# https://www.gnu.org/software/libtool/manual/html_node/Updating-version-info.html
m4_define([incremental_ver], [0])

m4_define([major_ver], m4_esyscmd([awk '/#define GDR_API_MAJOR_VERSION/ { print $3}' include/gdrapi.h| tr -d '\n']))
m4_define([minor_ver], m4_esyscmd([awk '/#define GDR_API_MINOR_VERSION/ { print $3}' include/gdrapi.h| tr -d '\n']))

m4_define([project_version], [major_ver.minor_ver])
m4_define([lt_cur], [m4_eval(major_ver + minor_ver)])
m4_define([lt_rev], [incremental_ver])
m4_define([lt_age], [minor_ver])
m4_define([version_info], [lt_cur:lt_rev:lt_age])

LT_PREREQ([2.4.6])

AC_PREREQ([2.69])
AC_INIT([gdrcopy], [project_version], [gpudirect@github.com])
AC_CONFIG_HEADERS([config.h])
AC_CONFIG_SRCDIR([src/gdrapi.c])
AC_CONFIG_AUX_DIR(config)
AC_CONFIG_MACRO_DIRS([config/m4])
AC_CANONICAL_TARGET
AM_INIT_AUTOMAKE([1.15 -Wall foreign tar-ustar silent-rules subdir-objects])
m4_ifdef([AM_SILENT_RULES], [AM_SILENT_RULES([yes])])

# Checks for programs.
AM_PROG_AR
AC_PROG_AWK
AC_PROG_CXX
AC_PROG_CC([gcc])
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL

LT_INIT

AC_SUBST(VERSION_INFO, [version_info])

AC_ARG_WITH(
    [cuda],
    [AC_HELP_STRING([--with-cuda@<:@=DIR@:>@],
                    [Build with cuda @<:@default=check@:>@])],
    [
        LDFLAGS="$LDFLAGS -L${withval}/lib64"
        CFLAGS="$CFLAGS -I${withval}/include"
        CXXFLAGS="$CXXFLAGS -I${withval}/include"
    ],
    []
)

AC_ARG_ENABLE(
	[test],
	[AC_HELP_STRING([--enable-test],
					[Build test programs @<:@default=no@:>@])],
	[enable_test=$enableval],
	[enable_test=no])

AC_ARG_ENABLE(
	[driver],
	[AC_HELP_STRING([--enable-driver@<:@=DIR@:>@],
					[Build the gdrdrv driver and install at the specified path @<:@default=yes@:>@])],
	[enable_driver=$enableval],
	[enable_driver=yes])

# Checks for header files.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h malloc.h memory.h netdb.h stddef.h stdint.h stdlib.h string.h sys/ioctl.h sys/socket.h sys/time.h unistd.h], [], [AC_MSG_FAILURE([Some of the above required headers are not found])])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_OFF_T
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_MMAP
AC_CHECK_FUNCS([memset munmap strerror strtol])

AC_CHECK_FUNC(clock_gettime, [], [
    AC_CHECK_LIB(rt, clock_gettime, [
        LDFLAGS="$LDFLAGS -lrt"
    ], [
        AC_MSG_FAILURE([clock_gettime is not found in the standard libraries nor in librt])
    ])
])

AS_IF(
	[test "x$enable_test" = "xyes"], [
	AC_MSG_NOTICE([Test is enabled])
	AC_CHECK_LIB([cuda], [cuMemAlloc], [], [AC_MSG_FAILURE([libcuda not found])])
	AC_CHECK_LIB([cudart], [cudaMalloc], [], [AC_MSG_FAILURE([libcudart not found])])
	AC_CHECK_HEADER([cuda.h], [], [AC_MSG_FAILURE([cuda.h not found])])
	AC_CHECK_HEADER([cuda_runtime_api.h], [], [AC_MSG_FAILURE([cuda_runtime_api.h not found])])
	PKG_CHECK_MODULES([CHECK], [check >= 0.9.8])
], [
	AC_MSG_NOTICE([Test is disabled])
])
AM_CONDITIONAL(TEST_ENABLE, test "x$enable_test" = "xyes")

AS_IF(
	[test "x$enable_driver" != "xno"], [
	AC_MSG_NOTICE([gdrdrv is enabled])
	AS_IF(
		[test "x$enable_driver" = "xyes"],
		[driver_path=$libdir],
		[driver_path=$enable_driver]
	)
	AC_SUBST(DRIVER_PATH, $driver_path)
], [
	AC_MSG_NOTICE([gdrdrv is disabled])
])
AM_CONDITIONAL(DRIVER_ENABLE, test "x$enable_driver" != "xno")

AS_IF(
    [expr match "x$target_cpu" "xpowerpc"], [arch=ppc],
    [expr match "x$target_cpu" "xx86"], [arch=x86],
    [AC_MSG_FAILURE([Compiling on unsupported platform!])]
)

AM_CONDITIONAL(ARCH_PPC, test "x$arch" = "xppc")
AM_CONDITIONAL(ARCH_X86, test "x$arch" = "xx86")

AC_CONFIG_FILES([Makefile src/Makefile src/gdrdrv/Makefile tests/Makefile])
AC_OUTPUT
