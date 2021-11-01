#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>
#define eps 1e-9

#pragma GCC optimize("Ofast", "unroll-loops")

pthread_mutex_t mutex;
long long ans = 0;

const double rand_min = -1.0;
const double rand_max =  1.0;

void* thread_func(void* tim) {

	long long t = *((long long*) tim);
    unsigned int mystate = 214786124; // time(NULL);

	double x, y;
	long long tmp_ans = 0;

	for (long long i = 0; i < t; ++i) {
		x = (rand_max - rand_min) * rand_r(&mystate) / (RAND_MAX + 1.0) + rand_min;
		y = (rand_max - rand_min) * rand_r(&mystate) / (RAND_MAX + 1.0) + rand_min;

		if (x * x + y * y - rand_max <= eps) tmp_ans += 1;
	}


	pthread_mutex_lock(&mutex);

	ans += tmp_ans;

	pthread_mutex_unlock(&mutex);

	pthread_exit(NULL);
}


int main(int argc, char const *argv[]) {
	
	int num_thread = atoi(argv[1]);
	long long num_toss = strtoll(argv[2], NULL, 10);

	#ifdef MYDEBUG
		printf("Now testing: thread = %d, toss = %lld\n\n", num_thread, num_toss);
	#endif

	pthread_t* thread_handles = (pthread_t*) malloc(num_thread * sizeof(pthread_t));
	long long* tims = (long long*) malloc(num_thread * sizeof(long long));
	pthread_mutex_init(&mutex, NULL);


	long thread;
	for (thread = 0; thread < num_thread; thread++) {
		tims[thread] = num_toss / num_thread;

		if (thread == num_thread - 1)
			tims[thread] += num_toss % num_thread;

		pthread_create(&thread_handles[thread], (pthread_attr_t*) NULL, thread_func, (void*) &tims[thread]);

		#ifdef MYDEBUG
			printf("Create thread = %ld, toss = %lld\n", thread, tims[thread]);
		#endif
	}
	for (thread = 0; thread < num_thread; thread++) {
		pthread_join(thread_handles[thread], NULL);
	}


	#ifdef MYDEBUG
		printf("\n=========================================\n");
		printf("\nAns = %lld, out of toss = %lld\n\n", ans, tims[thread]);
	#endif

	double pi_estimate = ans * 4.0 / (double) num_toss;
	printf("%.7lf\n", pi_estimate);

	pthread_mutex_destroy(&mutex);
	free(thread_handles);
	return 0;
}
