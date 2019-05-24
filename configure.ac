#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.69])
AC_INIT([gdrcopy], [1.0], [gpudirect@github.com])
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

AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
