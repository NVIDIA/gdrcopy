%{!?_release: %define _release 1}
%{!?CUDA: %define CUDA /usr/local/cuda}
%{!?GDR_VERSION: %define GDR_VERSION 2.0}
%{!?KVERSION: %define KVERSION %(uname -r)}
%{!?MODULE_LOCATION: %define MODULE_LOCATION /kernel/drivers/misc/}
%global debug_package %{nil}
%global krelver %(echo -n %{KVERSION} | sed -e 's/-/_/g')
%define MODPROBE %(if ( /sbin/modprobe -c | grep -q '^allow_unsupported_modules  *0'); then echo -n "/sbin/modprobe --allow-unsupported-modules"; else echo -n "/sbin/modprobe"; fi )
%define usr_src_dir /usr/src
%define old_driver_install_dir /lib/modules/%{KVERSION}/%{MODULE_LOCATION}
%global kmod kmod

%global kmod_install_script                                             \
echo "Start gdrcopy-kmod installation."                                 \
dkms add -m gdrdrv -v %{version} -q --rpm_safe_upgrade || :             \
                                                                        \
# Rebuild and make available for all installed kernel                   \
echo "Building and installing to all available kernels."                \
echo "This process may take a few minutes ..."                          \
for kver in $(ls -1d /lib/modules/* | cut -d'/' -f4)                    \
do                                                                      \
    dkms build -m gdrdrv -v %{version} -k ${kver} -q || :               \
    dkms install -m gdrdrv -v %{version} -k ${kver} -q --force || :     \
done                                                                    \
                                                                        \
/sbin/depmod -a %{KVERSION} &> /dev/null ||:                            \
%{MODPROBE} -rq gdrdrv||:                                               \
%{MODPROBE} gdrdrv||:                                                   \
                                                                        \
if ! ( /sbin/chkconfig --del gdrcopy > /dev/null 2>&1 ); then           \
   true                                                                 \
fi                                                                      \
                                                                        \
/sbin/chkconfig --add gdrcopy                                           \
                                                                        \
service gdrcopy start                                                    


%global daemon_reload_script                                            \
if [ -e /usr/bin/systemctl ]; then                                      \
    /usr/bin/systemctl daemon-reload                                    \
fi



Name:           gdrcopy
Version:        %{GDR_VERSION}
Release:        %{_release}%{?dist}
Summary:        GDRcopy library and companion kernel-mode driver    
Group:          System Environment/Libraries
License:        MIT
URL:            https://github.com/NVIDIA/gdrcopy
Source0:        %{name}-%{version}.tar.gz
BuildRequires:  gcc kernel-headers check-devel
Requires:       %{name}-%{kmod} check

# to get rid of libcuda/libcudart
AutoReqProv:    no

%package devel
Summary: The development files
Group: System Environment/Libraries
Requires: %{name} = %{version}-%{release}
BuildArch: noarch

%package %{kmod}
Summary: The kernel-mode driver
Group: System Environment/Libraries
Requires: dkms >= 1.00
Requires: bash
Release: %{_release}%{?dist}dkms
BuildArch: noarch
%if 0%{?rhel} >= 8
# Recommends tag is a weak dependency, whose support started in RHEL8.
# See https://access.redhat.com/documentation/en-us/red_hat_enterprise_linux/8/html/packaging_and_distributing_software/new-features-in-rhel-8_packaging-and-distributing-software#support-for-weak-dependencies_new-features-in-rhel-8.
Recommends: kmod-nvidia-latest-dkms
%endif

%description
GDRCopy, a low-latency GPU memory copy library and a kernel-mode driver, built on top of the 
NVIDIA GPUDirect RDMA technology.

%description devel
GDRCopy, a low-latency GPU memory copy library and a kernel-mode driver, built on top of the 
NVIDIA GPUDirect RDMA technology.

%description %{kmod}
Kernel-mode driver for GDRCopy.

%prep
%setup


%build
echo "building"
make -j CUDA=%{CUDA} config lib exes

%install
# Install gdrcopy library and tests
make install DESTDIR=$RPM_BUILD_ROOT prefix=%{_prefix} libdir=%{_libdir}

# Install gdrdrv src
mkdir -p $RPM_BUILD_ROOT%{usr_src_dir}
cp -r -a $RPM_BUILD_DIR/%{name}-%{version}/src/gdrdrv $RPM_BUILD_ROOT%{usr_src_dir}/gdrdrv-%{version}
cp -a $RPM_BUILD_DIR/%{name}-%{version}/dkms.conf $RPM_BUILD_ROOT%{usr_src_dir}/gdrdrv-%{version}

# Install gdrdrv service script
install -d $RPM_BUILD_ROOT/etc/init.d
install -m 0755 $RPM_BUILD_DIR/%{name}-%{version}/init.d/gdrcopy $RPM_BUILD_ROOT/etc/init.d

%post %{kmod}
if [ "$1" == "2" ] && [ -e "%{old_driver_install_dir}/gdrdrv.ko" ]; then
    echo "Old package is detected. Defer installation until after the old package is removed."

    # Prevent the uninstall scriptlet of the old package complaining about change in gdrcopy service
    %{daemon_reload_script}

    exit 0;
fi

# Prevent race with kmod-nvidia-latest-dkms triggerin
if [ ! -e "%{_localstatedir}/lib/rpm-state/gdrcopy-kmod/installed" ]; then
    %{kmod_install_script}
    mkdir -p %{_localstatedir}/lib/rpm-state/gdrcopy-kmod
    touch %{_localstatedir}/lib/rpm-state/gdrcopy-kmod/installed
fi


%preun %{kmod}
service gdrcopy stop||:
%{MODPROBE} -rq gdrdrv||:
if ! ( /sbin/chkconfig --del gdrcopy > /dev/null 2>&1 ); then
   true
fi              

# Remove all versions from DKMS registry
echo "Uninstalling and removing the driver."
echo "This process may take a few minutes ..."
dkms uninstall -m gdrdrv -v %{version} -q --all || :
dkms remove -m gdrdrv -v %{version} -q --all --rpm_safe_upgrade || :

# Clean up the weak-updates symlinks
find /lib/modules/*/weak-updates -name "gdrdrv.ko.*" -delete &> /dev/null || :
find /lib/modules/*/weak-updates -name "gdrdrv.ko" -delete &> /dev/null || :


%postun %{kmod}
%{daemon_reload_script}


%triggerpostun %{kmod} -- gdrcopy-kmod <= 2.1-1
%{kmod_install_script}


%triggerin %{kmod} -- kmod-nvidia-latest-dkms
if [ "$1" == "2" ] && [ -e "%{old_driver_install_dir}/gdrdrv.ko" ]; then
    echo "kmod-nvidia-latest-dkms is detected but defer installation because of the old gdrcopy-kmod package."
    exit 0;
fi

# Prevent race with post
if [ ! -e "%{_localstatedir}/lib/rpm-state/gdrcopy-kmod/installed" ]; then
    %{kmod_install_script}
    mkdir -p %{_localstatedir}/lib/rpm-state/gdrcopy-kmod
    touch %{_localstatedir}/lib/rpm-state/gdrcopy-kmod/installed
fi


%triggerun %{kmod} -- kmod-nvidia-latest-dkms
# This kmod package has only weak dependency with kmod-nvidia-latest-dkms, which is not enforced by RPM.
# Uninstalling kmod-nvidia-latest-dkms would not result in uninstalling this package.
# However, gdrdrv may prevent the removal of nvidia.ko.
# Hence, we rmmod gdrdrv before starting kmod-nvidia-latest-dkms uninstallation.
service gdrcopy stop||:
%{MODPROBE} -rq gdrdrv||:
service gdrcopy start > /dev/null 2>&1 ||:


%posttrans %{kmod}
# Cleaning up
if [ -e "%{_localstatedir}/lib/rpm-state/gdrcopy-kmod/installed" ]; then
    rm -f %{_localstatedir}/lib/rpm-state/gdrcopy-kmod/installed
fi


%clean
rm -rf $RPM_BUILD_DIR/%{name}-%{version}
[ "x$RPM_BUILD_ROOT" != "x" ] && rm -rf $RPM_BUILD_ROOT


%files
%{_prefix}/bin/copybw
%{_prefix}/bin/copylat
%{_prefix}/bin/sanity
%{_libdir}/libgdrapi.so.?.?
%{_libdir}/libgdrapi.so.?
%{_libdir}/libgdrapi.so


%files devel
%{_prefix}/include/gdrapi.h
%doc README.md


%files %{kmod}
%defattr(-,root,root,-)
/etc/init.d/gdrcopy
%{usr_src_dir}/gdrdrv-%{version}/gdrdrv.c
%{usr_src_dir}/gdrdrv-%{version}/gdrdrv.h
%{usr_src_dir}/gdrdrv-%{version}/Makefile
%{usr_src_dir}/gdrdrv-%{version}/nv-p2p-dummy.c
%{usr_src_dir}/gdrdrv-%{version}/dkms.conf


%changelog
* Mon Feb 01 2021 Pak Markthub <pmarkthub@nvidia.com> %{GDR_VERSION}-%{_release}
- Add support for ARM64.
- Update various information on README.
- Improve Makefile.
- Add multi-arch support.
- Handle removal of HAVE_UNLOCKED_IOCTL in Linux kernel v5.9 and later.
- Prevent dpkg package creation to unnecessarily compile gdrdrv.
- Improve gdr_open error message.
- Fix bug that prevents sanity from correctly summarizing failure.
- Add dkms support in kmod package.
- Handle the removal of kzfree in Linux kernel v5.10 and later.
- Improve small-size copy-to-mapping.
* Mon Jan 18 2021 Pak Markthub <pmarkthub@nvidia.com> 2.1-2
- Add DKMS support in gdrcopy-kmod.rpm
* Fri Jul 31 2020 Davide Rossetti <drossetti@nvidia.com> 2.1-1
- fix build problem on RHL8 kernels
- relax checks in gdrdrv to support multi-threading use cases
- fix fd leak in gdr_open()
* Mon Mar 02 2020 Davide Rossetti <drossetti@nvidia.com> 2.0-4
- Introduce copylat test application.
- Introduce basic_with_tokens and invalidation_fork_child_gdr_pin_parent_with_tokens sub-tests in sanity.
- Remove the dependency with libcudart.so.
- Clean up the code in the tests folder.
- Change the package maintainer to Davide Rossetti.
* Mon Sep 16 2019 Pak Markthub <pmarkthub@nvidia.com> 2.0-3
- Harden security in gdrdrv.
- Enable cached mappings in POWER9.
- Improve copy performance with unrolling in POWERPC.
- Creates _sanity_ unit test for testing the functionality and security.
- Consolidate basic and _validate_ into sanity unit test.
- Introduce compile time and runtime version checking in libgdrapi.
- Improve rpm packaging.
- Introduce deb packaging for the userspace library and the applications.
- Introduce dkms packaging for the gdrdrv driver.
- Rename gdr_copy_from/to_bar to gdr_copy_from/to_mapping.
- Update README
* Thu Jul 26 2018 Davide Rossetti <davide.rossetti@gmail.com> 1.4-2
- bumped minor version
* Fri Jun 29 2018 Davide Rossetti <davide.rossetti@gmail.com> 1.3-2
- a few bug fixes
* Mon Feb 13 2017 Davide Rossetti <davide.rossetti@gmail.com> 1.2-2
- package libgdrcopy.so as well
- add basic test
* Thu Sep 15 2016 Davide Rossetti <davide.rossetti@gmail.com> 1.2-1
- First version of RPM spec

