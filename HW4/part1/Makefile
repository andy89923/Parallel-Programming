hello: hello.cc
	mpicxx ./hello.cc -o mpi_hello && \
	mpirun -np 8 --hostfile hosts mpi_hello

test_hello: hello
	for i in 1...16; do 
		printf "\n\nTest %d\n" $${i}; 
		mpirun -np $${i} --hostfile hosts mpi_hello; 
		sleep 0.5; 
	done


clean:
	rm -f mpi_hello