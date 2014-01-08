#include <emmintrin.h>
#include <nmmintrin.h>
#include <stdio.h>
#include <string.h>
#include <omp.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

void printMatrix(float* matrix, int x, int y) {

    for (int i = 0; i < y; i++) {
        for (int j = 0; j < x; j++) {
        printf("%f ", *(matrix + j + y*i));
        }
        printf("\n");
    }
}

int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
		float* kernel)
{
	// the x coordinate of the kernel's center
	int kern_cent_X = (KERNX - 1)/2;
	// the y coordinate of the kernel's center
	int kern_cent_Y = (KERNY - 1)/2;


	// we want to flip the kernel first so that it doesn't happen in every single iteration of the loop. 
	int length = KERNX*KERNY;
	float flipped_kernel[length];
	for (int i = 0; i < length; i++) {
		flipped_kernel[length - (i+1)] = kernel[i];
	}

	// parallelization of the padded matrix
	float *padded_in = malloc((data_size_Y + KERNY)*(data_size_X + KERNX)*sizeof(float));
	
	#pragma omp parallel for firstprivate(data_size_X, data_size_Y)
	for (int m = 0; m < (data_size_Y + KERNY)*(data_size_X + KERNX); m++) {
		padded_in[m] = 0.0;
	}

	// input of the data into the padded matrix
	#pragma omp parallel for firstprivate(kern_cent_X, kern_cent_Y, data_size_X, padded_in, in)
	for(int i = 0; i < data_size_Y; i++) {
		for (int j = 0; j < data_size_X/32*32; j+=32) {
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 0 + kern_cent_X , _mm_loadu_ps(in+ j + 0  + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 4 + kern_cent_X , _mm_loadu_ps(in+ j + 4 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 8 + kern_cent_X , _mm_loadu_ps(in+ j + 8 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 12 + kern_cent_X , _mm_loadu_ps(in+ j + 12 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 16 + kern_cent_X , _mm_loadu_ps(in+ j + 16 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 20 + kern_cent_X , _mm_loadu_ps(in+ j + 20 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 24 + kern_cent_X , _mm_loadu_ps(in+ j + 24 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 28 + kern_cent_X , _mm_loadu_ps(in+ j + 28 + i *data_size_X));
		}
		
		for (int j = data_size_X/32*32; j < data_size_X/16*16; j+=16) {
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 0 + kern_cent_X , _mm_loadu_ps(in+ j + 0  + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 4 + kern_cent_X , _mm_loadu_ps(in+ j + 4 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 8 + kern_cent_X , _mm_loadu_ps(in+ j + 8 + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 12 + kern_cent_X , _mm_loadu_ps(in+ j + 12 + i *data_size_X));
		}

		for (int j = data_size_X/16*16; j < data_size_X/8*8; j+=8) {
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 0 + kern_cent_X , _mm_loadu_ps(in+ j + 0  + i *data_size_X));
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 4 + kern_cent_X , _mm_loadu_ps(in+ j + 4 + i *data_size_X));
		}

		for (int j = data_size_X/8*8; j < data_size_X/4*4; j+=4) {
			_mm_storeu_ps(padded_in + (i + kern_cent_Y)*(data_size_X + KERNX) + j + 0 + kern_cent_X , _mm_loadu_ps(in+ j + 0  + i *data_size_X));
		}

		for(int j = data_size_X/4*4; j < data_size_X; j++) {
			padded_in[(i + kern_cent_Y)*(data_size_X + KERNX) + j + kern_cent_X] = in[i*data_size_X + j];
		}
	}

	// parallelized calculation of the data and storage into output matrix
	#pragma omp parallel for firstprivate(data_size_X, data_size_Y, kern_cent_X, kern_cent_Y, flipped_kernel, padded_in)
	for(int y = 0; y < data_size_Y; y++) {

		for(int x = 0; x < (data_size_X)/32*32; x+=32) {

			__m128 sum1 = _mm_setzero_ps();
			__m128 sum2 = _mm_setzero_ps();
			__m128 sum3 = _mm_setzero_ps();
			__m128 sum4 = _mm_setzero_ps();
			__m128 sum5 = _mm_setzero_ps();
			__m128 sum6 = _mm_setzero_ps();
			__m128 sum7 = _mm_setzero_ps();
			__m128 sum8 = _mm_setzero_ps();

			for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){

					sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 0 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 4 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 8 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 12 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum5 = _mm_add_ps(sum5, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 16 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum6 = _mm_add_ps(sum6, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 20 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum7 = _mm_add_ps(sum7, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 24 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum8 = _mm_add_ps(sum8, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 28 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));

				}

			}
			_mm_storeu_ps((out + x + 0 + y*data_size_X), sum1);
			_mm_storeu_ps((out + x + 4 + y*data_size_X), sum2);
			_mm_storeu_ps((out + x + 8 + y*data_size_X), sum3);
			_mm_storeu_ps((out + x + 12 + y*data_size_X), sum4);
			_mm_storeu_ps((out + x + 16 + y*data_size_X), sum5);
			_mm_storeu_ps((out + x + 20 + y*data_size_X), sum6);
			_mm_storeu_ps((out + x + 24 + y*data_size_X), sum7);
			_mm_storeu_ps((out + x + 28 + y*data_size_X), sum8);

		}

		for(int x = data_size_X/32*32; x < data_size_X/16*16; x+=16) {

			__m128 sum1 = _mm_setzero_ps();
			__m128 sum2 = _mm_setzero_ps();
			__m128 sum3 = _mm_setzero_ps();
			__m128 sum4 = _mm_setzero_ps();

			for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){

					sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 0 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 4 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum3 = _mm_add_ps(sum3, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 8 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum4 = _mm_add_ps(sum4, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 12 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));

				}
			}
			_mm_storeu_ps((out + x + 0 + y*data_size_X), sum1);
			_mm_storeu_ps((out + x + 4 + y*data_size_X), sum2);
			_mm_storeu_ps((out + x + 8 + y*data_size_X), sum3);
			_mm_storeu_ps((out + x + 12 + y*data_size_X), sum4);

		}

		for(int x = data_size_X/16*16; x < data_size_X/8*8; x+=8) {

			__m128 sum1 = _mm_setzero_ps();
			__m128 sum2 = _mm_setzero_ps();

			for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
					sum1 = _mm_add_ps(sum1, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 0 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
					sum2 = _mm_add_ps(sum2, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + 4 + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
				}
			}
			_mm_storeu_ps((out + x + 0 + y*data_size_X), sum1);
			_mm_storeu_ps((out + x + 4 + y*data_size_X), sum2);
		}

		for(int x = data_size_X/8*8; x < data_size_X/4*4; x+=4) {
			__m128 sum = _mm_setzero_ps();
			for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
					sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps((flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX)), 
								_mm_loadu_ps(padded_in + x + kern_cent_X + i + (y + kern_cent_Y +j) * (data_size_X + KERNX))));
				}
			}
			_mm_storeu_ps((out + x + y*data_size_X), sum);
		}

		for(int x = data_size_X/4*4; x < data_size_X; x++) {
			float sum = 0;
			for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
					sum += flipped_kernel[(kern_cent_X + i) + (kern_cent_Y + j)*KERNX] * padded_in[(x+i + kern_cent_X) + (y + kern_cent_Y +j) * (data_size_X + KERNX)];
				}
			}
			out[x + y*data_size_X] = sum;
		}
	}

	free(padded_in);
	return 1;
} 
