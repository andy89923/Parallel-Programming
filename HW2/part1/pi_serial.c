#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#define eps 1e-9

int main(int argc, char const *argv[]) {
	
	long long num_toss = strtoll(argv[1], NULL, 10);

	printf("Number of toss: %lld\n", num_toss);

	srand(time(NULL));

	long long ans = 0;

	double rand_min = -1.0;
	double rand_max =  1.0;
	double x, y;
	for (int i = 0; i < num_toss; i++) {
		x = (rand_max - rand_min) * rand() / (RAND_MAX + 1.0) + rand_min;
		y = (rand_max - rand_min) * rand() / (RAND_MAX + 1.0) + rand_min;

		if (x * x + y * y - rand_max <= eps) ans += 1;
	}

	double pi_estimate = 4.0 * ans / (double) num_toss;

	printf("%.7lf\n", pi_estimate);
	return 0;
}