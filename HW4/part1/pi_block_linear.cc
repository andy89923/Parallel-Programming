#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
typedef long long lol;
#define eps 1e-9

#pragma GCC optimize("Ofast", "unroll-loops")

int main(int argc, char **argv) {
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	

	double rand_min = -1.0;
	double rand_max =  1.0;
	lol ans_sum = 0;

    if (world_rank > 0) {
        // TODO: handle workers
        lol num_toss;
        MPI_Recv(&num_toss, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        unsigned int seed = world_rank ^ time(NULL);
        lol* ans = (lol*) calloc(1, sizeof(lol));

        double x, y;
        for (int i = 0; i < num_toss; i++) {
            x = (rand_max - rand_min) * rand_r(&seed) / (RAND_MAX + 1.0) + rand_min;
            y = (rand_max - rand_min) * rand_r(&seed) / (RAND_MAX + 1.0) + rand_min;

            if (x * x + y * y - rand_max <= eps) *ans += 1;
        }
		// printf("%d %d %d\n", world_rank, num_toss, *ans);

        MPI_Send(ans, 1, MPI_LONG_LONG, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0) {
        // TODO: master

        lol num_toss = tosses / world_size;
        lol* trows = (lol*) calloc(world_size, sizeof(lol));
		
        for (int i = 1; i < world_size; i++) {
            trows[i] = num_toss;

            if (i == world_size - 1)
                trows[i] += tosses % world_size - 1;

            MPI_Send(&trows[i], 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD);
        }

		unsigned int seed = time(NULL);

		double x, y;
		for (int i = 0; i < num_toss; i++) {
			x = (rand_max - rand_min) * rand_r(&seed) / (RAND_MAX + 1.0) + rand_min;
			y = (rand_max - rand_min) * rand_r(&seed) / (RAND_MAX + 1.0) + rand_min;

			if (x * x + y * y - rand_max <= eps) ans_sum += 1;
		}	
    }
	
    if (world_rank == 0) {
        // TODO: process PI result
        lol tmp;

        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&tmp, 1, MPI_LONG_LONG, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ans_sum += tmp;

			// printf("%d %d\n", i, tmp);
        }
        pi_result = (double) 4.0 * ans_sum / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
