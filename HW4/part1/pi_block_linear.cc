#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>

int main(int argc, char **argv) {
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    // TODO: init MPI
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    if (world_rank > 0) {
        // TODO: handle workers
        int num_toss;
        MPI_Recv(&num_toss, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        int seed = world_rank ^ 31415926;
        srand(seed);
        
        int* ans = calloc(1, sizeof(int));

        double rand_min = -1.0;
        double rand_max =  1.0;
        double x, y;
        for (int i = 0; i < num_toss; i++) {
            x = (rand_max - rand_min) * rand() / (RAND_MAX + 1.0) + rand_min;
            y = (rand_max - rand_min) * rand() / (RAND_MAX + 1.0) + rand_min;

            if (x * x + y * y - rand_max <= eps) *ans += 1;
        }

        MPI_Send(ans, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    else if (world_rank == 0) {
        // TODO: master

        int num_toss = tosses / (world_size - 1);
        int* trows = calloc(world_size, sizeof(int));

        for (int i = 1; i < world_size; i++) {
            trows[i] = num_toss;

            if (i == world_size - 1)
                trows[i] += tosses % (world_size - 1);

            MPI_Send(&trows[i], 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    }

    if (world_rank == 0) {
        // TODO: process PI result
        int ans = 0, tmp;

        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&tmp, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            ans += tmp;
        }
        pi_result = (double) ans / tosses;

        // --- DON'T TOUCH ---
        double end_time = MPI_Wtime();
        printf("%lf\n", pi_result);
        printf("MPI running time: %lf Seconds\n", end_time - start_time);
        // ---
    }

    MPI_Finalize();
    return 0;
}
