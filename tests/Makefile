DESTBIN ?= 
CUDA ?= /usr/local/cuda
NVCC ?= $(CUDA)/bin/nvcc

# note that pkg-config respects PKG_CONFIG_PATH, which can be used
# to pick libcheck from a user-defined path
CHECK_INC  := $(shell pkg-config --cflags-only-I check)
CHECK_LIBS := $(shell pkg-config --libs check)

GDRAPI_INC := ../include
GDRAPI_SRC := ../src

CUDA_LIB := -L $(CUDA)/lib64 -L $(CUDA)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia -L $(CUDA)/lib64/stubs
CUDA_INC += -I $(CUDA)/include

CPPFLAGS := $(CUDA_INC) -I $(GDRAPI_INC) -I $(GDRAPI_SRC) -I $(CUDA)/include $(CHECK_INC)
LDFLAGS  := $(CUDA_LIB) -L $(CUDA)/lib64 -L $(GDRAPI_SRC)
COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
NVCCFLAGS ?=
LIBS     := -lcuda -lpthread -ldl -lgdrapi

CPP_SRCS := copybw.cpp sanity.cpp copylat.cpp apiperf.cpp
CU_SRCS  := pplat.cu
EXES := $(patsubst %.cpp,gdrcopy_%,$(CPP_SRCS)) $(patsubst %.cu,gdrcopy_%,$(CU_SRCS))

all: exes

exes: $(EXES)

common.o: common.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
copybw.o: copybw.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
sanity.o: sanity.cpp $(GDRAPI_INC)/gdrapi.h $(GDRAPI_SRC)/gdrapi_internal.h common.hpp
copylat.o: copylat.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
apiperf.o: apiperf.cpp $(GDRAPI_INC)/gdrapi.h common.hpp

gdrcopy_copybw: copybw.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

gdrcopy_sanity: sanity.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) $(CHECK_LIBS)

gdrcopy_copylat: copylat.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

gdrcopy_apiperf: apiperf.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

gdrcopy_pplat: pplat.o common.o
	$(NVCC)  -o $@ $^ $(LDFLAGS) -lgdrapi -lcuda

%.o: %.cu
	$(NVCC) -o $@ -c $^ $(LIBS) $(CPPFLAGS) $(NVCCFLAGS)

clean:
	rm -f *.o $(EXES) *~ core.*

install: exes
	@ echo "installing exes in $(DESTBIN)..." && \
	mkdir -p $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx gdrcopy_copybw -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx gdrcopy_copylat -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx gdrcopy_apiperf -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx gdrcopy_sanity -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx gdrcopy_pplat -t $(DESTBIN)
	cd $(DESTBIN) && \
	ln -sf gdrcopy_copybw copybw && \
	ln -sf gdrcopy_copylat copylat && \
	ln -sf gdrcopy_apiperf apiperf && \
	ln -sf gdrcopy_sanity sanity

.PHONY: clean all exes install
