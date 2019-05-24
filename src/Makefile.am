AM_CPPFLAGS  = -I$(srcdir)
AM_CPPFLAGS += -I$(srcdir)/gdrdrv

lib_LTLIBRARIES = libgdrapi.la
libgdrapi_la_SOURCES = gdrapi.c 
libgdrapi_la_LDFLAGS = -version-info 0:0:0 @LIBOBJS@

noinst_LTLIBRARIES = libmemcpy_avx.la
libmemcpy_avx_la_SOURCES = memcpy_avx.c
libmemcpy_avx_la_CFLAGS = -mavx

noinst_LTLIBRARIES += libmemcpy_sse.la
libmemcpy_sse_la_SOURCES = memcpy_sse.c
libmemcpy_sse_la_CFLAGS = -msse

noinst_LTLIBRARIES += libmemcpy_sse41.la
libmemcpy_sse41_la_SOURCES = memcpy_sse41.c
libmemcpy_sse41_la_CFLAGS = -msse4.1
