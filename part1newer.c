#include <emmintrin.h>
#include <nmmintrin.h>
#include <string.h>
#include <stdio.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

void printMatrix(float* matrix, int x, int y) {
	for (int i = 0; i < y; i++) {
		for (int j = 0; j < x; j++) {
			printf("%f \t", matrix + j + ij);
			printf("\n");
		}
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


	float *padded_in = calloc((data_size_Y+2)*(data_size_X+2), sizeof(float));
	int d_x = data_size_X + 2;
	
	int size_f = sizeof(float)*data_size_X;
	float* pad_1 = padded_in + 1;

	for (int i = 1; i <= data_size_Y ; i++) {
	    memcpy(pad_1 + i*d_x, in + (data_size_X*(i-1)), size_f);
	}

	//__m128 vpad;
	//__m128 vfk;
	//float final[4];

	for(int y = 1; y <= data_size_Y/4*4; y+=4) {
	    for(int x = 1; x <= data_size_X/4*4; x+=4) {
		int xy_location = (x-1)+ (y-1)*data_size_X;
		__m128 res = _mm_setzero_ps();
			for(int i = -kern_cent_X; i <= kern_cent_X; i++) {
		    	for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
					out[xy_location] += 
			    		flipped_kernel[(kern_cent_X + i) + (kern_cent_Y + j)*KERNY] * padded_in[(x+i) + (y+j) * d_x];
			    		// vfk = _mm_loadu_ps(flipped_kernel + (kern_cent_X + i) + (kern_cent_Y + j) * KERNY);
			    		// vpad = _mm_loadu_ps(padded_in + (x+i) + (y+j) * d_x);
			    		// res = _mm_add_ps(res, _mm_mul_ps(vfk, vpad));
		    	}
				// _mm_storeu_ps(final, res);
				//	out[xy_location] = final[0] + final[1] + final[2] + final[3];
				//
				//
				// (insert tail case here)
			}
	  	}
	}
	
	free(padded_in);
	return 1;
} 
