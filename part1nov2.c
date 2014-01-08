#include <emmintrin.h>
#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
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

/*
	float *padded_in = calloc((data_size_Y+ KERNY/2*2)*(data_size_X + KERNX/2*2), sizeof(float));
	int d_x = data_size_X + KERNX/2*2;
	
	int size_f = sizeof(float)*data_size_X;
	float* pad_1 = padded_in + KERNX/2;

	for (int i = KERNY/2; i < data_size_X + KERNY/2; i++) {
	    memcpy(pad_1 + i*d_x, in + (data_size_X*(i-KERNX/2)), size_f);
	}
	
*/
	float *padded_in = calloc((data_size_Y + KERNY)*(data_size_X + KERNX), sizeof(float));
	for(int i = 0; i < data_size_Y; i++){
		memcpy(padded_in + KERNX/2 + (i + KERNY/2)*(data_size_X + KERNX), in + i*data_size_X, sizeof(float)*data_size_X);
	}

	__m128 sum;
	float final[4];

	for(int y = 0; y < data_size_Y; y++) {
	    for(int x = 0; x < (data_size_X)/4*4; x+=4) {
		int xy_location = x + y*data_size_X;
		sum = _mm_setzero_ps();
	         for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
		 for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
			sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel + i + kern_cent_X + (kern_cent_Y+j)*KERNX), 
				_mm_loadu_ps(padded_in + (x + KERNX/2 + i) + (y + KERNY/2 +j) * (data_size_X + KERNX))));
			
		    }

		}
		_mm_storeu_ps( (__m128*)(out + xy_location), sum);

	     }
	//should be naive implementation
	    for(int x = data_size_X/4*4; x < data_size_X; x++) {
		int xy_location = x + y*data_size_X;
		sum = _mm_setzero_ps();
	         for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
		 for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
			out[xy_location] += 
			    flipped_kernel[(kern_cent_X + i) + (kern_cent_Y + j)*KERNX] * padded_in[(x+i + KERNX/2) + (y + KERNY/2 +j) * (data_size_X + KERNX)];
		    }
		}
	     }
	}


/*
	for(int y = KERNY/2; y <= data_size_Y + KERNY/2; y++) {
	    for(int x = KERNX/2; x <= data_size_X + KERNX/2; x++) {
		int xy_location = (x-KERNX/2)+ (y-KERNY/2)*data_size_X;
	         for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
		 for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
			out[xy_location] += 
			    flipped_kernel[(kern_cent_X + i) + (kern_cent_Y + j)*KERNX] * padded_in[(x+i) + (y+j) * d_x];

		    }
		}

	     }
	}

*/


	/*__m128 sum;
	float final[4];


	for(int y = 1; y <= data_size_Y; y++) {
	    for(int x = 1; x < (data_size_X-1)/4*4 + 1; x+=4) {
		int xy_location = (x-1)+ (y-1)*data_size_X;
		sum = _mm_setzero_ps();
	         for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
		 for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
			sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel + i + kern_cent_X), _mm_loadu_ps(padded_in + (x+i) + (y+j) * d_x)));
		    }
		}
		_mm_storeu_ps( (__m128*)(out + xy_location), sum);

	     }
	    for(int x = (data_size_X - 1)/4*4 + 1; x < data_size_X; x++) {
		int xy_location = (x-1)+ (y-1)*data_size_X;
		sum = _mm_setzero_ps();
	         for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
		 for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
			sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel + i + kern_cent_X), _mm_loadu_ps(padded_in + (x+i) + (y+j) * d_x)));
		    }
		}
		_mm_storeu_ps((__m128*)(out + xy_location) , sum);

	     }
	}
*/


	//printf("padded_in \n");
   	//printMatrix(padded_in, data_size_X+KERNX, data_size_Y+KERNY);
	
	//printf("kernel \n"); 
    	//printMatrix(flipped_kernel, 3, 3);
	
	//printf("out \n");
	//printMatrix(out, data_size_X, data_size_Y);


/*
	__m128 sum;
	float final[4];
	for(int y = 1; y <= data_size_Y; y++) {
	    	for(int x = 1; x <= data_size_X/4*4; x+=4) {
		sum = _mm_setzero_ps();
		int xy_location = (x-1)+ (y-1)*(data_size_X);
 	       sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+0) , _mm_loadu_ps(padded_in +(x-1) + (y-1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+1) , _mm_loadu_ps(padded_in + x + (y-1)*d_x )));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+2) , _mm_loadu_ps(padded_in +(x+1) + (y-1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+3) , _mm_loadu_ps(padded_in + (x-1) + y*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+4) , _mm_loadu_ps(padded_in + x + y*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+5) , _mm_loadu_ps(padded_in + (x+1) + y*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+6) , _mm_loadu_ps(padded_in +(x-1) + (y+1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+7) , _mm_loadu_ps(padded_in +x + (y+1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+8) , _mm_loadu_ps(padded_in + (x+1) + (y+1)*d_x)));
	       _mm_storeu_ps(xy_location + out, sum);

	     }

	    	for(int x = data_size_X/4*4 + 1; x <= data_size_X; x+=4) {
		sum = _mm_setzero_ps();
		int xy_location = (x-1)+ (y-1)*(data_size_X);
 	       sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+0) , _mm_loadu_ps(padded_in +(x-1) + (y-1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+1) , _mm_loadu_ps(padded_in + x + (y-1)*d_x )));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+2) , _mm_loadu_ps(padded_in +(x+1) + (y-1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+3) , _mm_loadu_ps(padded_in + (x-1) + y*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+4) , _mm_loadu_ps(padded_in + x + y*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+5) , _mm_loadu_ps(padded_in + (x+1) + y*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+6) , _mm_loadu_ps(padded_in +(x-1) + (y+1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+7) , _mm_loadu_ps(padded_in +x + (y+1)*d_x)));
               sum = _mm_add_ps(sum, _mm_mul_ps(_mm_load1_ps(flipped_kernel+8) , _mm_loadu_ps(padded_in + (x+1) + (y+1)*d_x)));
	       _mm_storeu_ps(xy_location + out, sum);

	     }
	}
*/

	free(padded_in);
	return 1;

} 
