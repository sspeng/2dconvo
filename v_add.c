#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define ARRAY_SIZE 10000000
#define REPEAT     100


// Edit this function
void v_add(double* x, double* y, double* z) {
	#pragma omp parallel
	{
		// NAIVE
		// for(int i=0; i<ARRAY_SIZE; i++) {
		// 	z[i] = x[i] + y[i];
		// }

		//int thread_ID = omp_get_thread_num();
		//int numthreads = omp_get_num_threads();
		//for(int i=thread_ID; i<ARRAY_SIZE; i+=numthreads) {
		// 	z[i] = x[i] + y[i];
		//}

		int thread_ID = omp_get_thread_num();
		int numthreads = omp_get_num_threads();
		int chunk = ARRAY_SIZE/numthreads;
		//int count = 0;
		//#pragma omp for
		for (int i = thread_ID * chunk; i < thread_ID*chunk + chunk; i++) {
			z[i] = x[i] + y[i];
		}
		if (thread_ID == numthreads - 1) {
			for(int i=ARRAY_SIZE - ARRAY_SIZE%chunk; i<ARRAY_SIZE ; i++) {
		 		z[i] = x[i] + y[i];
			}
		}
	}
}


double* gen_array(int n) {
	double* array = (double*) malloc(n*sizeof(double));
	for(int i=0; i<n; i++)
		array[i] = drand48();
	return array;
}

// Double check if it is correct
int verify(double* x, double* y) {
	double *z_v_add = (double*) malloc(ARRAY_SIZE*sizeof(double));
	double *z_oracle = (double*) malloc(ARRAY_SIZE*sizeof(double));
	v_add(x, y, z_v_add);
	for(int i=0; i<ARRAY_SIZE; i++)
		z_oracle[i] = x[i] + y[i];
	for(int i=0; i<ARRAY_SIZE; i++)
		if(z_oracle[i] != z_v_add[i])
			return 0;	
	return 1;
}


int main() {
	// Generate input vectors and destination vector
	double *x = gen_array(ARRAY_SIZE);
	double *y = gen_array(ARRAY_SIZE);
	double *z = (double*) malloc(ARRAY_SIZE*sizeof(double));

	// Test framework that sweeps the number of threads and times each run
	double start_time, run_time;
	int num_threads = omp_get_max_threads();	
	for(int i=1; i<=num_threads; i++) {
		omp_set_num_threads(i);		
		start_time = omp_get_wtime();
		for(int j=0; j<REPEAT; j++)
			v_add(x,y,z);
		run_time = omp_get_wtime() - start_time;

	  // verify that output of v_add is correct
		if(!verify(x,y)) {
			printf("v_add does not match oracle\n");
			return -1;
		}
  	printf(" %d thread(s) took %f seconds\n",i,run_time);
	}
  return 0;
}
