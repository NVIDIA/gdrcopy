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

prefix      ?= /usr/local
exec_prefix ?= $(prefix)
libdir      ?= $(exec_prefix)/lib
bindir      ?= $(exec_prefix)/bin
includedir  ?= $(prefix)/include

DESTDIR := $(abspath $(DESTDIR))
DESTLIB = $(DESTDIR)$(libdir)
DESTBIN = $(DESTDIR)$(bindir)
DESTINC = $(DESTDIR)$(includedir)

CUDA ?= /usr/local/cuda

LIB_MAJOR_VER ?= $(shell awk '/\#define GDR_API_MAJOR_VERSION/ { print $$3 }' include/gdrapi.h | tr -d '\n')
LIB_MINOR_VER ?= $(shell awk '/\#define GDR_API_MINOR_VERSION/ { print $$3 }' include/gdrapi.h | tr -d '\n')

GDRAPI_ARCH := $(shell ./config_arch)
GDRAPI_INC := ../include

LIB_VER:=$(LIB_MAJOR_VER).$(LIB_MINOR_VER)
LIB_BASENAME:=libgdrapi.so
LIB_DYNAMIC=$(LIB_BASENAME).$(LIB_VER)
LIB_SONAME=$(LIB_BASENAME).$(LIB_MAJOR_VER)

ARCH := $(shell uname -m)
OS := linux
BUILD_DIR := build
INSTALL_ROOT := $(BUILD_DIR)/install

all: config driver lib exes

version:
	@ echo "$(LIB_VER)"

config:
	@ echo "GDRAPI_ARCH=$(GDRAPI_ARCH)"

driver:
	cd src/gdrdrv && \
	$(MAKE) $(MAKE_PARAMS)

lib:
	cd src && \
	$(MAKE) LIB_MAJOR_VER=$(LIB_MAJOR_VER) LIB_MINOR_VER=$(LIB_MINOR_VER)

exes: lib
	cd tests && \
	$(MAKE) CUDA=$(CUDA)

install: lib_install exes_install

lib_install: lib
	@ echo "installing in $(DESTLIB) $(DESTINC)..." && \
	mkdir -p $(DESTLIB) && \
	install -D -v -m 755 src/$(LIB_DYNAMIC) -t $(DESTLIB) && \
	mkdir -p $(DESTINC) && \
	install -D -v -m 664 include/* -t $(DESTINC); \
	cd $(DESTLIB); \
	ln -sf $(LIB_DYNAMIC) $(LIB_SONAME); \
	ln -sf $(LIB_SONAME) $(LIB_BASENAME);

exes_install: exes
	cd tests && $(MAKE) install DESTBIN=$(DESTBIN)


drv_install: driver
	cd src/gdrdrv && \
	$(MAKE) install

clean:
	cd tests && \
	$(MAKE) clean
	cd src && \
	$(MAKE) clean
	cd src/gdrdrv && \
	$(MAKE) clean

dist: dist-clean dist-prepare dist-lib dist-tests dist-driver
	rm -rf $(INSTALL_ROOT)

dist-prepare:
	mkdir -p $(BUILD_DIR)
	mkdir -p $(INSTALL_ROOT)

dist-lib: lib
	mkdir -p $(INSTALL_ROOT)/libgdrapi/lib64
	mkdir -p $(INSTALL_ROOT)/libgdrapi/include
	cp src/$(LIB_DYNAMIC) $(INSTALL_ROOT)/libgdrapi/lib64/
	cd $(INSTALL_ROOT)/libgdrapi/lib64 && \
	ln -sf $(LIB_DYNAMIC) $(LIB_SONAME) && \
	ln -sf $(LIB_SONAME) $(LIB_BASENAME)
	cp include/*.h $(INSTALL_ROOT)/libgdrapi/include/
	cp LICENSE $(INSTALL_ROOT)/libgdrapi/
	cd $(INSTALL_ROOT) && \
	tar czf ../libgdrapi-$(OS)-$(ARCH)-$(LIB_VER).tar.gz libgdrapi/

dist-tests: exes
	mkdir -p $(INSTALL_ROOT)/gdrcopy-tests/bin
	cp tests/gdrcopy_* $(INSTALL_ROOT)/gdrcopy-tests/bin/
	cp LICENSE $(INSTALL_ROOT)/gdrcopy-tests/
	cd $(INSTALL_ROOT) && \
	tar czf ../gdrcopy-tests-$(OS)-$(ARCH)-$(LIB_VER).tar.gz gdrcopy-tests/

dist-driver:
	cd src/gdrdrv && make clean
	mkdir -p $(INSTALL_ROOT)/gdrdrv/src
	cp src/gdrdrv/* $(INSTALL_ROOT)/gdrdrv/src/
	cp LICENSE $(INSTALL_ROOT)/gdrdrv/
	cd $(INSTALL_ROOT) && \
	tar czf ../gdrdrv-$(OS)-$(ARCH)-$(LIB_VER).tar.gz gdrdrv/

dist-clean:
	rm -rf $(INSTALL_ROOT)
	rm -rf $(BUILD_DIR)

.PHONY: driver clean all lib exes lib_install drv_install exes_install install dist dist-prepare dist-lib dist-tests dist-driver dist-clean

