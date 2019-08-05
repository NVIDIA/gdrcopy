LIB_MAJOR_VER ?= 1
LIB_MINOR_VER ?= 4

GDRAPI_ARCH ?= $(shell ../config_arch)
GDRAPI_INC := ../include

CPPFLAGS := -I $(GDRAPI_INC) -I gdrdrv/ -D GDRAPI_ARCH=$(GDRAPI_ARCH)
LDFLAGS  := 
COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -lpthread -ldl

LIB_VER:=$(LIB_MAJOR_VER).$(LIB_MINOR_VER)
LIB_BASENAME:=libgdrapi.so
LIB_DYNAMIC=$(LIB_BASENAME).$(LIB_VER)
LIB_SONAME=$(LIB_BASENAME).$(LIB_MAJOR_VER)
LIB:=$(LIB_DYNAMIC)

LIBSRCS := gdrapi.c
ifeq ($(GDRAPI_ARCH),X86)
LIBSRCS += memcpy_avx.c memcpy_sse.c memcpy_sse41.c
endif

LIBOBJS := $(LIBSRCS:.c=.o)

all: config lib

config:
	@ echo "GDRAPI_ARCH=$(GDRAPI_ARCH)"

lib: $(LIB)

#static
#$(LIB): $(LIB)($(LIBOBJS))
#dynamic
$(LIBOBJS): CFLAGS+=-fPIC
$(LIB): $(LIBOBJS)
	$(CC) -shared -Wl,-soname,$(LIB_SONAME) -o $@ $^
	ldconfig -n $(PWD)
	ln -sf $(LIB_DYNAMIC) $(LIB_SONAME)
	ln -sf $(LIB_SONAME) $(LIB_BASENAME)

# special-cased to finely tune the arch option
memcpy_avx.o: memcpy_avx.c
	$(COMPILE.c) -mavx -o $@ $^

memcpy_sse.o: memcpy_sse.c
	$(COMPILE.c) -msse -o $@ $^

memcpy_sse41.o: memcpy_sse41.c
	$(COMPILE.c) -msse4.1 -o $@ $^

gdrapi.o: gdrapi.c $(GDRAPI_INC)/gdrapi.h 

clean:
	rm -f *.o $(EXES) lib*.so* *~ core.*

.PHONY: clean all lib 
