TARGET := pi.out
CPP := pi.c

CXX := gcc
CFLAGS := -std=gnu99 -Ofast
THREAD := -pthread

all: $(CPP)
	$(CXX) $(CPP) -o $(TARGET) $(CFLAGS) $(THREAD)

clean:
	rm -f *.out *.o

serial:
	$(CXX) pi_serial.c -o test.o $(CFLAGS) $(THREAD)
	./test.o 100000000
	rm test.o

debug: $(CPP)
	$(CXX) $(CPP) -o $(TARGET) $(CFLAGS) $(THREAD) -DMYDEBUG