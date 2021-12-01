#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/types.h>
#include <unistd.h>
typedef long long lol;
#define eps 1e-9

const double rand_min = -1.0;
const double rand_max =  1.0;

void rand_toss(lol& ans, lol num_toss, unsigned int seed) {
    double x, y;
    for (lol i = 0; i < num_toss; i++) {
        x = (rand_max - rand_min) * rand_r(&seed) / (RAND_MAX + 1.0) + rand_min;
        y = (rand_max - rand_min) * rand_r(&seed) / (RAND_MAX + 1.0) + rand_min;

        if (x * x + y * y - rand_max <= eps) ans += 1;
    }
}

int main(int argc, char **argv) {
    // --- DON'T TOUCH ---
    MPI_Init(&argc, &argv);
    double start_time = MPI_Wtime();
    double pi_result;
    long long int tosses = atoi(argv[1]);
    int world_rank, world_size;
    // ---

    MPI_Win win;

    // TODO: MPI init
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);


    lol num_toss = tosses / world_size;
    if (world_rank == world_size - 1)
        num_toss += tosses % world_size;

    unsigned int seed = time(NULL) ^ world_rank;

    lol ans = 0;
    rand_toss(ans, num_toss, seed);


    lol ans_sum;
    if (world_rank == 0) {
        // Master
        MPI_Alloc_mem(sizeof(lol), MPI_INFO_NULL, &ans_sum);
        ans_sum = ans;
        MPI_Win_create(&ans_sum, sizeof(lol), sizeof(lol), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    }
    else {
        // Workers

        MPI_Win_create(NULL, 0, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &win);

        MPI_Win_lock(MPI_LOCK_EXCLUSIVE, 0, 0, win);
        MPI_Accumulate(&ans, 1, MPI_LONG_LONG, 0, 0, 1, MPI_LONG_LONG, MPI_SUM, win);
        MPI_Win_unlock(0, win);
    }
    MPI_Win_free(&win);

    if (world_rank == 0) {
        // TODO: handle PI result

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
