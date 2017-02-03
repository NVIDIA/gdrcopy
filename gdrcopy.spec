%{!?_release: %define _release 1}
%{!?CUDA: %define CUDA /usr/local/cuda}
%{!?KVERSION: %define KVERSION %(uname -r)}
%global krelver %(echo -n %{KVERSION} | sed -e 's/-/_/g')
%define MODPROBE %(if ( /sbin/modprobe -c | grep -q '^allow_unsupported_modules  *0'); then echo -n "/sbin/modprobe --allow-unsupported-modules"; else echo -n "/sbin/modprobe"; fi )
%define driver_install_dir /lib/modules/%{KVERSION}/extra
%global kmod kmod
#modules-%{krelver}


Name:           gdrcopy
Version:        1.2
Release:        %{_release}%{?dist}
Summary:        GDRcopy library and companion kernel-mode driver    
Group:          System Environment/Libraries
License:        MIT
URL:            https://github.com/NVIDIA/gdrcopy
Source0:        %{name}-%{version}.tar.gz
BuildRequires:  gcc kernel-headers
Requires:       %{name}-%{kmod} 

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
export KVER=%{KVERSION}
echo $KVER
make %{?_smp_mflags} CUDA=%{CUDA} KVER=%{KVERSION} all


%install
%{__mkdir_p} $RPM_BUILD_ROOT%{_libdir}
%{__mkdir_p} $RPM_BUILD_ROOT%{_prefix}/include
%{__make} PREFIX=$RPM_BUILD_ROOT%{_prefix} DESTLIB=$RPM_BUILD_ROOT%{_libdir} lib_install
install -d $RPM_BUILD_ROOT%{_prefix}/include
#install -m 0755 gdrapi.h $RPM_BUILD_ROOT%{_prefix}/include/gdrapi.h
install -Dpm 755 copybw $RPM_BUILD_ROOT%{_prefix}/bin/copybw
install -Dpm 755 validate $RPM_BUILD_ROOT%{_prefix}/bin/validate

# Install gdrdrv service script
install -d $RPM_BUILD_ROOT/etc/init.d
install -m 0755 $RPM_BUILD_DIR/%{name}-%{version}/init.d/gdrcopy $RPM_BUILD_ROOT/etc/init.d


%{__mkdir_p} $RPM_BUILD_ROOT%{driver_install_dir}
%{__cp} $RPM_BUILD_DIR/%buildsubdir/gdrdrv/gdrdrv.ko $RPM_BUILD_ROOT%{driver_install_dir}



%post
/sbin/depmod -a
%{MODPROBE} -rq gdrdrv||:
%{MODPROBE} gdrdrv||:

if ! ( /sbin/chkconfig --del gdrcopy > /dev/null 2>&1 ); then
   true
fi              

/sbin/chkconfig --add gdrcopy

%preun
%{MODPROBE} -rq gdrcopy
if ! ( /sbin/chkconfig --del gdrcopy > /dev/null 2>&1 ); then
   true
fi              


%clean
rm -rf $RPM_BUILD_DIR/%{name}-%{version}
[ "x$RPM_BUILD_ROOT" != "x" ] && rm -rf $RPM_BUILD_ROOT


%files
%{_prefix}/bin/copybw
%{_prefix}/bin/validate
%{_libdir}/libgdrapi.so.?.?
%{_libdir}/libgdrapi.so.?
%{_libdir}/libgdrapi.so
/etc/init.d/gdrcopy


%files devel
%{_libdir}/libgdrapi.so
%{_prefix}/include/gdrapi.h
%doc README.md


%files %{kmod}
%defattr(-,root,root,-)
%{driver_install_dir}/gdrdrv.ko


%changelog
* Thu Sep 15 2016 Davide Rossetti <davide.rossetti@gmail.com> 1.2-1
- First version of RPM spec

