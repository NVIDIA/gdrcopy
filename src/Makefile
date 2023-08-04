# Copyright (c) 2019-2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

LIB_MAJOR_VER ?= $(shell awk '/\#define GDR_API_MAJOR_VERSION/ { print $$3 }' ../include/gdrapi.h | tr -d '\n')
LIB_MINOR_VER ?= $(shell awk '/\#define GDR_API_MINOR_VERSION/ { print $$3 }' ../include/gdrapi.h | tr -d '\n')

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
	PATH=/sbin:/usr/sbin:$$PATH; ldconfig -n $(PWD)
	ln -sf $(LIB_DYNAMIC) $(LIB_SONAME)
	ln -sf $(LIB_SONAME) $(LIB_BASENAME)

# special-cased to finely tune the arch option
memcpy_avx.o: memcpy_avx.c
	$(COMPILE.c) -mavx -o $@ $^

memcpy_sse.o: memcpy_sse.c
	$(COMPILE.c) -msse -o $@ $^

memcpy_sse41.o: memcpy_sse41.c
	$(COMPILE.c) -msse4.1 -o $@ $^

gdrapi.o: gdrapi.c $(GDRAPI_INC)/gdrapi.h gdrapi_internal.h gdrdrv/gdrdrv.h

clean:
	rm -f *.o $(EXES) lib*.so* *~ core.*

.PHONY: clean all lib 
