DESTBIN ?= 
CUDA ?= /usr/local/cuda

GDRAPI_INC := ../include
GDRAPI_SRC := ../src

CUDA_LIB := -L $(CUDA)/lib64 -L $(CUDA)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia
CUDA_INC += -I $(CUDA)/include

CPPFLAGS := $(CUDA_INC) -I $(GDRAPI_INC) -I $(GDRAPI_SRC) -I $(CUDA)/include
LDFLAGS  := $(CUDA_LIB) -L $(CUDA)/lib64 -L $(GDRAPI_SRC)
COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -lcuda -lpthread -ldl -lgdrapi

SRCS := copybw.cpp sanity.cpp
EXES := $(SRCS:.cpp=)

all: exes

exes: $(EXES)

common.o: common.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
copybw.o: copybw.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
sanity.o: sanity.cpp $(GDRAPI_INC)/gdrapi.h $(GDRAPI_SRC)/gdrapi_internal.h common.hpp

copybw: copybw.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) -lrt

sanity: sanity.o common.o
	$(LINK.cc)  -o $@ $^ $(LIBS) `pkg-config --libs check`

clean:
	rm -f *.o $(EXES) *~ core.*

install: exes
	@ echo "installing exes in $(DESTBIN)..." && \
	mkdir -p $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx copybw -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx sanity -t $(DESTBIN)

.PHONY: clean all exes install
