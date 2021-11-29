hello: hello.cc
	mpicxx ./hello.cc -o mpi_hello && \
	mpirun -np 8 --hostfile hosts mpi_hello

test_hello: hello
	for i in 8 16; do printf "\n\nTest %d\n" $${i}; mpirun -np $${i} --hostfile hosts mpi_hello; sleep 0.5; done

linear:
	mpicxx pi_block_linear.cc -o pi_block_linear; mpirun -np 4 --hostfile hosts pi_block_linear 1000000000

clean:
	rm -f mpi_hello
