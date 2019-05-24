AM_CPPFLAGS  = -I$(srcdir)

bin_PROGRAMS = basic
basic_SOURCES = basic.cpp
basic_LDADD = -lgdrapi -lcuda -lcudart

bin_PROGRAMS += copybw
copybw_SOURCES = copybw.cpp
copybw_LDADD = -lgdrapi -lcuda -lcudart

bin_PROGRAMS += validate
validate_SOURCES = copybw.cpp
validate_LDADD = -lgdrapi -lcuda -lcudart

