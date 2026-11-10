#/usr/bin/make
SRC = $(DESTDIR)/usr/src
SHARE = $(DESTDIR)/usr/share/$(NAME)-dkms
MODULES_LOAD_DIR = $(DESTDIR)/usr/lib/modules-load.d
UDEV_RULES_DIR = $(DESTDIR)/usr/lib/udev/rules.d
MODPROBE_DIR = $(DESTDIR)/usr/lib/modprobe.d
LIBEXEC_DIR = $(DESTDIR)/usr/libexec/gdrcopy

all:

clean:

install: install-config

#source tree
ifeq ("$(wildcard $(NAME)-$(VERSION))", "$(NAME)-$(VERSION)")
	install -d "$(SRC)"
	cp -a $(NAME)-$(VERSION) $(SRC)

	# sets 0755 for dirs, 0644 for files
	chmod a-wx+rX,u+w -R "$(SRC)/$(NAME)-$(VERSION)"

	# set u+x for all files under the scripts folder
	chmod u+x -R "$(SRC)/$(NAME)-$(VERSION)/scripts"
endif

#tarball, possibly with binaries
ifeq ("$(wildcard $(NAME)-$(VERSION).dkms.tar.gz)", "$(NAME)-$(VERSION).dkms.tar.gz")
	install -d "$(SHARE)"
	install -m 644 $(NAME)-$(VERSION).dkms.tar.gz "$(SHARE)"
endif

#postinst, only if we are supporting legacy mode
ifeq ("$(wildcard common.postinst)", "common.postinst")
	install -d "$(SHARE)"
	install -m 755 $(PREFIX)/usr/lib/dkms/common.postinst $(SHARE)/postinst
endif

#boot loading, device-node permissions, and proprietary load hooks
install-config:
	install -d "$(MODULES_LOAD_DIR)"
	install -m 644 gdrdrv.conf "$(MODULES_LOAD_DIR)/gdrdrv.conf"
	install -d "$(UDEV_RULES_DIR)"
	install -m 644 70-gdrdrv.rules "$(UDEV_RULES_DIR)/70-gdrdrv.rules"
	install -d "$(MODPROBE_DIR)"
	install -m 644 gdrdrv-modprobe.conf "$(MODPROBE_DIR)/gdrdrv.conf"
	install -d "$(LIBEXEC_DIR)"
	install -m 755 gdrdrv_devnode.sh "$(LIBEXEC_DIR)/gdrdrv_devnode.sh"
