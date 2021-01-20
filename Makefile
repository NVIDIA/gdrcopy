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
	install -D -v -m u=rw,g=rw,o=r src/$(LIB_DYNAMIC) -t $(DESTLIB) && \
	mkdir -p $(DESTINC) && \
	install -D -v -m u=rw,g=rw,o=r include/* -t $(DESTINC); \
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

.PHONY: driver clean all lib exes lib_install drv_install exes_install install

