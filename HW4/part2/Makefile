all: matmul.cc
	mpicxx -O3 -Ofast -ffast-math main.cc matmul.cc -o matmul 

run:
	make; mpirun -np 2 --hostfile hosts matmul < test.in

clean:
	rm -f matmul
