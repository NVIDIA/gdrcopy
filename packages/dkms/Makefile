#/usr/bin/make
SRC = $(DESTDIR)/usr/src
SHARE = $(DESTDIR)/usr/share/$(NAME)-dkms

all:

clean:

install:

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
