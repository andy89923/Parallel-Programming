hello: hello.cc
	mpicxx ./hello.cc -o mpi_hello && \
	mpirun -np 8 --hostfile hosts --map-by node mpi_hello

test_hello: hello
	for i in 8 16; do printf "\n\nTest %d\n" $${i}; mpirun -np $${i} --hostfile hosts --map-by node mpi_hello; sleep 0.5; done

linearb:
	mpicxx pi_block_linear.cc -o pi_block_linear; mpirun -np 4 --hostfile hosts --map-by node pi_block_linear 1000000000

test_linearb:
	mpirun -np  2 --hostfile hosts --map-by node pi_block_linear 1000000000
	mpirun -np  4 --hostfile hosts --map-by node pi_block_linear 1000000000
	mpirun -np  8 --hostfile hosts --map-by node pi_block_linear 1000000000
	mpirun -np 12 --hostfile hosts --map-by node pi_block_linear 1000000000
	mpirun -np 16 --hostfile hosts --map-by node pi_block_linear 1000000000

linear:
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 4 --hostfile hosts --map-by node pi_nonblock_linear 1000000000

test_linear:
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 2 --hostfile hosts --map-by node pi_nonblock_linear 1000000000
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 4 --hostfile hosts --map-by node pi_nonblock_linear 1000000000
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 8 --hostfile hosts --map-by node pi_nonblock_linear 1000000000
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 12 --hostfile hosts --map-by node pi_nonblock_linear 1000000000
	mpicxx pi_nonblock_linear.cc -o pi_nonblock_linear; mpirun -np 16 --hostfile hosts --map-by node pi_nonblock_linear 1000000000

tree:
	mpicxx pi_block_tree.cc -o pi_block_tree; mpirun -np 4 --hostfile hosts --map-by node pi_block_tree 1000000000	

test_tree:
	mpirun -np  2 --hostfile hosts --map-by node pi_block_tree 1000000000
	mpirun -np  4 --hostfile hosts --map-by node pi_block_tree 1000000000
	mpirun -np  8 --hostfile hosts --map-by node pi_block_tree 1000000000
	mpirun -np 16 --hostfile hosts --map-by node pi_block_tree 1000000000

gather:
	mpicxx pi_gather.cc -o pi_gather; mpirun -np 4 --hostfile hosts --map-by node pi_gather 1000000000

test_gather:
	mpirun -np  2 --hostfile hosts --map-by node pi_gather 1000000000
	mpirun -np  4 --hostfile hosts --map-by node pi_gather 1000000000
	mpirun -np  8 --hostfile hosts --map-by node pi_gather 1000000000
	mpirun -np 12 --hostfile hosts --map-by node pi_gather 1000000000
	mpirun -np 16 --hostfile hosts --map-by node pi_gather 1000000000

reduce:
	mpicxx pi_reduce.cc -o pi_reduce; mpirun -np 4 --hostfile hosts --map-by node pi_reduce 1000000000

test_reduce:
	mpirun -np  2 --hostfile hosts --map-by node pi_reduce 1000000000
	mpirun -np  4 --hostfile hosts --map-by node pi_reduce 1000000000
	mpirun -np  8 --hostfile hosts --map-by node pi_reduce 1000000000
	mpirun -np 12 --hostfile hosts --map-by node pi_reduce 1000000000
	mpirun -np 16 --hostfile hosts --map-by node pi_reduce 1000000000

clean:
	rm -f mpi_hello pi_block_linear pi_block_tree pi_nonblock_linear pi_gather pi_reduce
