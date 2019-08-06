DESTBIN ?= 
CUDA ?= /usr/local/cuda

GDRAPI_INC := ../include
GDRAPI_SRC := ../src

CUDA_LIB := -L $(CUDA)/lib64 -L $(CUDA)/lib -L /usr/lib64/nvidia -L /usr/lib/nvidia
CUDA_INC += -I $(CUDA)/include

CPPFLAGS := $(CUDA_INC) -I $(GDRAPI_INC) -I $(CUDA)/include
LDFLAGS  := $(CUDA_LIB) -L $(CUDA)/lib64 -L $(GDRAPI_SRC)
COMMONCFLAGS := -O2
CFLAGS   += $(COMMONCFLAGS)
CXXFLAGS += $(COMMONCFLAGS)
LIBS     := -lcudart -lcuda -lpthread -ldl -lgdrapi

SRCS := basic.cpp validate.cpp copybw.cpp
EXES := $(SRCS:.cpp=)

all: exes

exes: $(EXES)

validate.o: validate.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
basic.o: basic.cpp $(GDRAPI_INC)/gdrapi.h common.hpp
copybw.o: copybw.cpp $(GDRAPI_INC)/gdrapi.h common.hpp

basic: basic.o
	$(LINK.cc)  -o $@ $^ $(LIBS)

validate: validate.o
	$(LINK.cc)  -o $@ $^ $(LIBS)

copybw: copybw.o
	$(LINK.cc)  -o $@ $^ $(LIBS)

clean:
	rm -f *.o $(EXES) *~ core.*

install: exes
	@ echo "installing exes in $(DESTBIN)..." && \
	mkdir -p $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx basic -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx copybw -t $(DESTBIN) && \
	install -D -v -m u=rwx,g=rx,o=rx validate -t $(DESTBIN)

.PHONY: clean all exes install
