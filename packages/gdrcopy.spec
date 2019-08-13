%{!?_release: %define _release 3}
%{!?CUDA: %define CUDA /usr/local/cuda}
%{!?GDR_VERSION: %define GDR_VERSION 2.0}
%{!?KVERSION: %define KVERSION %(uname -r)}
%{!?MODULE_LOCATION: %define MODULE_LOCATION /kernel/drivers/misc/}
%global debug_package %{nil}
%global krelver %(echo -n %{KVERSION} | sed -e 's/-/_/g')
%define MODPROBE %(if ( /sbin/modprobe -c | grep -q '^allow_unsupported_modules  *0'); then echo -n "/sbin/modprobe --allow-unsupported-modules"; else echo -n "/sbin/modprobe"; fi )
%define driver_install_dir /lib/modules/%{KVERSION}/%{MODULE_LOCATION}
%global kmod kmod
#modules-%{krelver}


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
# alternatives, not working on RH6 
#%filter_from_provides /libcuda\\.so.*$/d
#%global __provides_exclude ^libcuda\\.so.*$

%package devel
Summary: The development files
Group: System Environment/Libraries
Requires: %{name} = %{version}-%{release}

%package %{kmod}
Summary: The kernel-mode driver
Group: System Environment/Libraries
#Requires: %{name} = %{version}-%{release}

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
make -j CUDA=${CUDA}

%install
make install DESTDIR=$RPM_BUILD_ROOT PREFIX=%{_prefix}
make drv_install DESTDIR=$RPM_BUILD_ROOT 

# Install gdrdrv service script
install -d $RPM_BUILD_ROOT/etc/init.d
install -m 0755 $RPM_BUILD_DIR/%{name}-%{version}/init.d/gdrcopy $RPM_BUILD_ROOT/etc/init.d

%post %{kmod}
/sbin/depmod -a
%{MODPROBE} -rq gdrdrv||:
%{MODPROBE} gdrdrv||:

if ! ( /sbin/chkconfig --del gdrcopy > /dev/null 2>&1 ); then
   true
fi              

/sbin/chkconfig --add gdrcopy

/usr/bin/systemctl start gdrcopy

%preun %{kmod}
/usr/bin/systemctl stop gdrcopy
%{MODPROBE} -rq gdrdrv
if ! ( /sbin/chkconfig --del gdrcopy > /dev/null 2>&1 ); then
   true
fi              

%postun %{kmod}
/usr/bin/systemctl daemon-reload


%clean
rm -rf $RPM_BUILD_DIR/%{name}-%{version}
[ "x$RPM_BUILD_ROOT" != "x" ] && rm -rf $RPM_BUILD_ROOT


%files
%{_prefix}/bin/copybw
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
%{driver_install_dir}/gdrdrv.ko


%changelog
* Fri Aug 16 2019 Pak Markthub <pmarkthub@nvidia.com> 2.0-3
- Harden security in gdrdrv.
- Enable cached mappings in POWER9.
- Improve copy performance with unrolling in POWERPC.
- Creates sanity unit test for testing the functionality and security.
- Consolidate basic and validate into sanity unit test.
- Introduce compile time and runtime version checking in libgdrapi.
- Improve rpm packaging.
- Introduce deb packaging for the userspace library and the applications.
- Introduce dkms packaging for the gdrdrv driver.
- Rename gdr_copy_from/to_bar to gdr_copy_from/to_mapping.
* Thu Jul 26 2018 Davide Rossetti <davide.rossetti@gmail.com> 1.4-2
- bumped minor version
* Fri Jun 29 2018 Davide Rossetti <davide.rossetti@gmail.com> 1.3-2
- a few bug fixes
* Mon Feb 13 2017 Davide Rossetti <davide.rossetti@gmail.com> 1.2-2
- package libgdrcopy.so as well
- add basic test
* Thu Sep 15 2016 Davide Rossetti <davide.rossetti@gmail.com> 1.2-1
- First version of RPM spec
