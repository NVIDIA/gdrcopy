AM_CPPFLAGS  = -I$(srcdir)
AM_CPPFLAGS += -I$(srcdir)/../include
AM_CPPFLAGS += -I$(srcdir)/../src
LDADD = -lgdrapi -lcuda -lcudart -L${top_builddir}/src/.libs/

bin_PROGRAMS = sanity
sanity_SOURCES = sanity.cpp
sanity_CFLAGS = @CHECK_CFLAGS@
sanity_LDADD = $(LDADD) @CHECK_LIBS@

bin_PROGRAMS += copybw
copybw_SOURCES = copybw.cpp

