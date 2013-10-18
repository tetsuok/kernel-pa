CXX = g++

#OPT = -O0 -g
OPT = -O3

CFLAGS =  -I. -Wall $(OPT) -Wno-sign-compare -Wconversion -Wcast-align -Wcast-qual
LDFLAGS =

PROGRAMS = \
	kernel_pa \
	kernel_pa_classify

OBJS = \
	learner.o \
	tokenizer.o

default: all

all: $(PROGRAMS)

kernel_pa: kernel_pa.o $(OBJS)
	$(CXX) -o $@ kernel_pa.o $(OBJS) $(LDFLAGS)

kernel_pa_classify: kernel_pa_classify.o $(OBJS)
	$(CXX) -o $@ kernel_pa_classify.o $(OBJS) $(LDFLAGS)

clean:
	-rm -f *.o $(PROGRAMS)

.cc.o:
	$(CXX) $(CFLAGS) -c $< -o $@
