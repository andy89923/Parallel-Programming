SHELL := /bin/bash

TARGET := pi.out
CPP := pi.c

CXX := gcc -std=gnu99
CFLAGS := -Ofast -O3 -mavx2
THREAD := -pthread

TESTCOMMAND := ./pi.out 3 100000000; ./pi.out 4 100000000

all: $(CPP)
	$(CXX) $(CPP) -o $(TARGET) $(CFLAGS) $(THREAD)

CTT := g++-10
# CTT := clang++-11

test:
	$(CTT) $(CFLAGS) $(CPP) -o $(TARGET) $(THREAD)
	time ($(TESTCOMMAND))

clean:
	rm -f *.out *.o

serial:
	$(CXX) pi_serial.c -o test.o $(CFLAGS) $(THREAD)
	./test.o 100000000
	rm test.o

debug: $(CPP)
	$(CXX) $(CPP) -o $(TARGET) $(CFLAGS) $(THREAD) -DMYDEBUG
