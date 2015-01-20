# Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
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

ifeq ($(src),)
	src := $$PWD
endif
NVIDIA_SRC_DIR := $(shell $(src)/path_to_p2p_header.sh)

ifneq ($(KERNELRELEASE),)

kver_major:=$(shell echo $(KERNELRELEASE) | awk -F '.' '// { print $$2;}' )
obj-m := gdrdrv.o
#EXTRA_CFLAGS += -Werror-implicit-function-declaration -Wno-format
EXTRA_CFLAGS += -I $(NVIDIA_SRC_DIR)

else

KDIR ?= /lib/modules/$(shell uname -r)/build
#REL := $(subst ., , $(shell lsb_release -rs))
REL := $(subst ., , $(subst -, , $(shell uname -r)))
REL_MAJOR  := $(word 1,$(REL))
REL_MEDIUM := $(word 2,$(REL))
REL_MINOR  := $(word 3,$(REL))

#NVIDIA_SRC_DIR := $(shell $$PWD/path_to_p2p_header.sh)

all: nvidia_src_dir linksyms default

nvidia_src_dir:
	@ [ -e $(src)/path_to_p2p_header.sh ] || { echo "can't exec path_to_p2p_header.sh"; exit 1; }; \
	echo "src dir: $(NVIDIA_SRC_DIR)"

linksyms:
	@ rm -f rm Module.symvers; \
	echo "kernel versions is: $(REL_MAJOR).$(REL_MEDIUM).$(REL_MINOR)"; \
        if [ $(REL_MAJOR) -gt 2 ]; then \
		echo "assuming this is RHEL 7.x"; \
		ln -sf Module.symvers.nvidia_on_rhel7.x Module.symvers; \
	elif [ $(REL_MINOR) -gt 18 ]; then \
		echo "assuming this is RHEL 6.x"; \
		ln -sf Module.symvers.nvidia_on_rhel6.x Module.symvers; \
	else \
		echo "assuming this is RHEL 5.x"; \
		ln -sf Module.symvers.nvidia_on_rhel5.x Module.symvers; \
	fi

default:
	@ $(MAKE) -C $(KDIR) M=$$PWD  modules

install:
	$(MAKE) -C $(KDIR) M=$$PWD  modules_install

help:
	$(MAKE) -C $(KDIR) M=$$PWD help

clean:
	rm -rf *.o *.ko* *.mod.* .*.cmd modules.order .tmp_versions/ *~ core .depend TAGS 

TAGS:
	find $(KERNELDIR) -follow -name \*.h -o -name \*.c  |xargs etags

.PHONY: clean all help install default linksyms nvidia_src_dir

endif
