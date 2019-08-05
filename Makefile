PREFIX ?= /usr/local
DESTLIB ?= $(PREFIX)/lib64
CUDA ?= /usr/local/cuda

GDRAPI_ARCH := $(shell ./config_arch)
GDRAPI_INC := ../include

LIB_MAJOR_VER:=1
LIB_MINOR_VER:=4

LIB_VER:=$(LIB_MAJOR_VER).$(LIB_MINOR_VER)
LIB_BASENAME:=libgdrapi.so
LIB_DYNAMIC=$(LIB_BASENAME).$(LIB_VER)
LIB_SONAME=$(LIB_BASENAME).$(LIB_MAJOR_VER)

all: config driver lib exes

config:
	@ echo "GDRAPI_ARCH=$(GDRAPI_ARCH)"

driver:
	cd src/gdrdrv && \
	$(MAKE) $(MAKE_PARAMS)

lib:
	cd src && \
	$(MAKE) LIB_MAJOR_VER=$(LIB_MAJOR_VER) LIB_MINOR_VER=$(LIB_MINOR_VER)

exes:
	cd tests && \
	$(MAKE) CUDA=$(CUDA)

install: lib_install #drv_install

lib_install:
	@ echo "installing in $(PREFIX)..." && \
	install -D -v -m u=rw,g=rw,o=r src/$(LIB_DYNAMIC) -t $(DESTLIB) && \
	install -D -v -m u=rw,g=rw,o=r include/* -t $(PREFIX)/include/; \
	cd $(DESTLIB); \
	ln -sf $(LIB_DYNAMIC) $(LIB_SONAME); \
	ln -sf $(LIB_SONAME) $(LIB_BASENAME);

drv_install:
	cd src/gdrdrv && \
	$(MAKE) install

clean:
	cd tests && \
	$(MAKE) clean
	cd src && \
	$(MAKE) clean
	cd src/gdrdrv && \
	$(MAKE) clean

.PHONY: driver clean all lib exes lib_install install

