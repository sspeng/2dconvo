#include <emmintrin.h>
#include <string.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.
int conv2D(float* in, float* out, int data_size_X, int data_size_Y,
                    float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    // we want to flip the kernel first so that it doesn't happen in every single iteration of the loop. 
    int flipped_kernel[KERNX*KERNY];
    for (int i = 0; i < KERNX*KERNY; i++) {
	flipped_kernel[KERNX*KERNY - (i+1)] = kernel[i];
    }

    // main convolution loop
/*	for(int x = 0; x < data_size_X; x++){ // the x coordinate of the output location we're focusing on
		for(int y = 0; y < data_size_Y; y++){ // the y coordinate of theoutput location we're focusing on
			for(int i = -kern_cent_X; i <= kern_cent_X; i++){ // kernel unflipped x coordinate
				for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){ // kernel unflipped y coordinate
					// only do the operation if not out of bounds
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						//Note that the kernel is flipped
						out[x+y*data_size_X] += 
								kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}
		}	
	}
	return 1;
}*/

    //int padded_in[(data_size_Y+2)*(data_size_X+2)];
	
	float *padded_in = calloc((data_size_Y+2)*(data_size_X+2), sizeof(float));

	for (int i = 1; i <= data_size_Y ; i++) {
	    memcpy(padded_in + i*(data_size_X+2) + 1, in + (data_size_X*(i-1)), sizeof(float)*data_size_X);
	}
	
	/* ARACELI
	for(int y = 1; y <= data_size_Y; y++) {
           for(int x = 1; x <= data_size_X; x++) {
               for(int j = -kern_cent_X; j <= kern_cent_X; j++){
                   for(int i = -kern_cent_Y; i <= kern_cent_Y; i++){
                       out[(x-1)+(y-1)*data_size_X] +=
                               flipped_kernel[KERNY* (kern_cent_X + j) + (kern_cent_Y + i)] * padded_in[(x+i) + (y+j)*(data_size_X+2)];
                   }
               }
            }
	}
	*/

	__m128i resval = _mm_setzero_si128();
	int temp[4] = {0,0,0,0};

	for(int y = 1; y <= data_size_Y/4*4; y+=4) {
           for(int x = 1; x <= data_size_X/4*4; x+=4) {
               	for(int j = -kern_cent_Y; j <= kern_cent_Y; j++){
			       	for(int i = -kern_cent_X; i <= kern_cent_X; i++){
						__m128i vect = _mm_loadu_si128((__m128i*)(padded_in[(x+j) + (y+i)*(data_size_X+2)]));
						__m128i res = _mm_mul_ps((__m128i*)(flipped_kernel[KERNY* (kern_cent_X + i) + (kern_cent_Y + j)]), 
												(__m128i*)((kern_cent_Y + j)] * padded_in[(x+j) + (y+i)*(data_size_X+2)]));
						_mm_storeu_si128((__m128i*)temp, res);
					for (int z = 0; z < 4; z++) {
						out[(x-1)+(y-1)*data_size_X] += *(temp + z);
					}
                }
            }
			for (int g = data_size_X/4*4; x < data_size_X; g++) {
				for (int h = data_size_Y/4*4; x < data_size_Y; h++) {
					out[(x-1)+(y-1)*data_size_X] +=
                        flipped_kernel[KERNY* (kern_cent_X + j) + (kern_cent_Y + i)] * padded_in[(g+i) + (h+j)*(data_size_X+2)];
				}
			}
        }
	}
	return 1;
}
