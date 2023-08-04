# Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

NVIDIA_SRC_DIR ?= $(shell { find /usr/src/kernel-modules/nvidia-* /usr/src/nvidia-* -name "nv-p2p.h" -print -quit | xargs dirname || echo "NVIDIA_DRIVER_MISSING"; } 2>/dev/null)

ifneq ($(KERNELRELEASE),)

kver_major:=$(shell echo $(KERNELRELEASE) | awk -F '.' '// { print $$2;}' )

obj-m := nv-p2p-dummy.o
obj-m += gdrdrv.o

ccflags-y += -I$(NVIDIA_SRC_DIR)

else

KVER ?= $(shell uname -r)
MODULES_DIR := /lib/modules/$(KVER)
KDIR := $(MODULES_DIR)/build
MODULE_SUBDIR ?= /kernel/drivers/misc/
MODULE_DESTDIR := $(MODULES_DIR)/$(MODULE_SUBDIR)
DEPMOD := /sbin/depmod

REL := $(subst ., , $(subst -, , $(shell uname -r)))
REL_MAJOR  := $(word 1,$(REL))
REL_MEDIUM := $(word 2,$(REL))
REL_MINOR  := $(word 3,$(REL))

all: build

build:
	@ echo "Picking NVIDIA driver sources from NVIDIA_SRC_DIR=$(NVIDIA_SRC_DIR). If that does not meet your expectation, you might have a stale driver still around and that might cause problems."
	@ $(MAKE) -C $(KDIR) $(MAKE_PARAMS) M=$(PWD) modules

install: build
	[ -d $(DESTDIR)/$(MODULE_DESTDIR) ] || mkdir -p $(DESTDIR)/$(MODULE_DESTDIR)
	cp gdrdrv.ko $(DESTDIR)/$(MODULE_DESTDIR)
	if [ ! -n "$(DESTDIR)" ]; then $(DEPMOD) -r -ae $(KVER); fi

help:
	$(MAKE) -C $(KDIR) M=$$PWD help

clean:
	rm -rf *.o .*.o.d *.ko* *.mod.* .*.cmd Module.symvers modules.order .tmp_versions/ *~ core .depend TAGS .cache.mk  *.mod

TAGS:
	find $(KERNELDIR) -follow -name \*.h -o -name \*.c  |xargs etags

.PHONY: clean all help install default linksyms nvidia_src_dir build

endif
