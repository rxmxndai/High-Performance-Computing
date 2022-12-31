#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime_api.h>



//compile with c++ lodepng file
//nvcc boxBlur.cu lodepng.cpp

__global__ void boxBlur(unsigned char *ImageInput, unsigned char * ImageOuput, int width, int height){

	int filter[] = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};

	int red=0;
	int green=0;
	int blue=0;
	int transperency=0;

	int index = blockDim.x * blockIdx.x + threadIdx.x;


	if(index==0){
		filter[0] = index;
		filter[1] = index+1;
		filter[2] = index+width;
		filter[3] = index+width+1;

	}
	else if(index>0 && index<(width-1)){
		filter[0] = index;
		filter[1] = index+1;
		filter[2] = index-1;
		filter[3] = index+width;
		filter[4] = 1+index+width;
		filter[5] = index+width-1;
	}
	else if (index==(width-1)){
		filter[0] = index;
		filter[1] = index-1;
		filter[2] = index+width;
		filter[3] = index+width-1;
	}
	else if(((index > width-1 && index< (height*width)-width) && ((index+1) % width ==0))){
		filter[0] = index;
		filter[1] = index-1;
		filter[2] = index-width;
		filter[3] = index-width-1;
		filter[4] = index+width;
		filter[5] = index+width-1;
	}
	else if (index==((height*width)-1)){
		filter[0] = index;
		filter[1] = index-1;
		filter[2] = index-width-1;
		filter[3] = index-width;
	}
	else if(index>((height*width)-width) && index < (height*width)){
		filter[0] = index;
		filter[1] = index+1;
		filter[2] = index-1;
		filter[3] = index-width;
		filter[4] = index-width-1;
		filter[5] = index-width+1;
	}
	else if(index==(height*width)-width){
		filter[0] = index;
		filter[1] = index+1;
		filter[2] = index-width;
		filter[3] = index-width+1;
	}
	else if((index>width-1 &&index<(height*width)-(2*width+1))&&index % width ==0){
		filter[0] = index;
		filter[1] = index+1;
		filter[2] = index+width;
		filter[3] = index+width+1;
		filter[4] = index-width;
		filter[5] = index-width+1;

	}
	else{
		filter[0] = index;
		filter[1] = index+1;
		filter[2] = index-1;
		filter[3] = index+width;
		filter[4] = index+width+1;
		filter[5] = index+width-1;
		filter[6] = index-width;
		filter[7] = index-width+1;
		filter[8] = index-width-1;
	}



	int pixel = index*4;
	int c=0;
	for (int index=0; index<sizeof(filter)/sizeof(filter[0]); index++){
		if(filter[index] != NULL){
			red += ImageInput[filter[index]*4];
			green += ImageInput[filter[index]*4+1];
			blue += ImageInput[filter[index]*4+2];
			c++;
		}
	}
	

	red = red/c;
	green = green/c;
	blue = blue/c;
	transperency = ImageInput[index*4+3];


	ImageOuput[pixel] = red;
	ImageOuput[1+pixel] = green;
	ImageOuput[2+pixel] = blue;
	ImageOuput[3+pixel] = transperency;
	
}



int main () {
    // png decode variables
    unsigned char *img;
    unsigned int height, width;

    // lodepng decode
    lodepng_decode32_file(&img, &width, &height, "hck.png");

	printf("Width: %d\nHeight: %d\n", width, height);

    // image pixel's values RGBT and its space on memory
    int totalValues = height * width * 4;
    int totalSpace = totalValues * sizeof(unsigned char);

    // dynamic memory allocation for image data in host
    unsigned char *hostImageInput;
    hostImageInput = (unsigned char *) malloc(totalSpace);
    unsigned char *hostImageOutput;
    hostImageOutput = (unsigned char *) malloc(totalSpace);


    // PUT IMAGE DATA INSIDE host input image array
    for (int index=0; index<totalValues; index++) {
        hostImageInput[index]  = img[index];
    }

    // declare memory pointers for GPU (device)
    unsigned char *deviceImageInput;
    unsigned char *deviceImageOutput;


    // allocate memory for gpu
    cudaMalloc( (void **) &deviceImageInput, totalSpace);
    cudaMalloc( (void **) &deviceImageOutput, totalSpace);


    // allocate memory in gpu with image data
    cudaMemcpy(deviceImageInput, hostImageInput, totalSpace, cudaMemcpyHostToDevice);


    // invoke kernel function that runs on device and blur the image with 3x3 matrix approach
    boxBlur<<<height, width>>>(deviceImageInput, deviceImageOutput, width, height);

    // copy device image data (blurred) to host image array
    cudaMemcpy(hostImageOutput, deviceImageOutput, totalSpace, cudaMemcpyDeviceToHost);
	
    // encode output image data to new image
	lodepng_encode32_file("cudaBlur.png", hostImageOutput, width, height);


    // free dynamic allocation
    free(img);
    free(hostImageInput);
    free(hostImageOutput);
    cudaFree(deviceImageInput);
    cudaFree(deviceImageOutput);

    return 0;
}