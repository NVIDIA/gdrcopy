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

AC_PREREQ([2.69])
AC_INIT([gdrcopy], [project_version], [gpudirect@github.com])
AC_CONFIG_SRCDIR([src/gdrapi.c])
AC_CONFIG_AUX_DIR(config)
AM_INIT_AUTOMAKE([1.10 -Wall foreign subdir-objects])
AC_CONFIG_MACRO_DIRS([config/m4])

# Checks for programs.
AM_PROG_AR
AC_PROG_AWK
AC_PROG_CXX
AC_PROG_CC
AC_PROG_INSTALL
AC_PROG_LN_S
AC_PROG_MAKE_SET
AC_PROG_LIBTOOL

LT_INIT

AC_SUBST(VERSION_INFO, [version_info])

# Checks for libraries.
# FIXME: Replace `main' with a function in `-lcuda':
#AC_CHECK_LIB([cuda], [main])
# FIXME: Replace `main' with a function in `-lcudart':
#AC_CHECK_LIB([cudart], [main])
# FIXME: Replace `main' with a function in `-ldl':
#AC_CHECK_LIB([dl], [main])
# FIXME: Replace `main' with a function in `-lpthread':
#AC_CHECK_LIB([pthread], [main])

# Checks for header files.
#AC_CHECK_HEADERS([arpa/inet.h fcntl.h malloc.h memory.h netdb.h stddef.h stdint.h stdlib.h string.h sys/ioctl.h sys/socket.h sys/time.h unistd.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_C_INLINE
AC_TYPE_OFF_T
AC_TYPE_SIZE_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_FUNC_MMAP
AC_CHECK_FUNCS([clock_gettime memset munmap strerror strtol])

AC_CONFIG_FILES([Makefile src/Makefile tests/Makefile])
AC_OUTPUT
