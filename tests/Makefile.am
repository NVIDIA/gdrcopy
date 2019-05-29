AM_CPPFLAGS  = -I$(srcdir)
AM_CPPFLAGS += -I$(srcdir)/../include
LDADD = -lgdrapi -lcuda -lcudart -L${top_builddir}/src/.libs/

bin_PROGRAMS = basic
basic_SOURCES = basic.cpp

bin_PROGRAMS += copybw
copybw_SOURCES = copybw.cpp

bin_PROGRAMS += validate
validate_SOURCES = copybw.cpp

