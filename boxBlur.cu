#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime_api.h>



//compile with c++ lodepng file
//nvcc boxBlur.cu lodepng.cpp

__global__ void boxBlur(unsigned char *ImageInput, unsigned char * ImageOuput, int width, int height){

	int x[]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};

	int red=0;
	int green=0;
	int blue=0;
	int transperency=0;

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i==0){
		x[0]=i;
		x[1]=i+1;
		x[2]=i+width;
		x[3]=i+width+1;

	}
	else if(i>0 && i<(width-1)){
		x[0]=i;
		x[1]=i+1;
		x[2]=i-1;
		x[3]=i+width;
		x[4]=1+i+width;
		x[5]=i+width-1;
	}
	else if (i==(width-1)){
		x[0]=i;
		x[1]=i-1;
		x[2]=i+width;
		x[3]=i+width-1;
	}
	else if(((i > width-1 && i< (height*width)-width) && ((i+1) % width ==0))){
		x[0]=i;
		x[1]=i-1;
		x[2]=i-width;
		x[3]=i-width-1;
		x[4]=i+width;
		x[5]=i+width-1;
	}
	else if (i==((height*width)-1)){
		x[0]=i;
		x[1]=i-1;
		x[2]=i-width-1;
		x[3]=i-width;
	}
	else if(i>((height*width)-width) && i < (height*width)){
		x[0]=i;
		x[1]=i+1;
		x[2]=i-1;
		x[3]=i-width;
		x[4]=i-width-1;
		x[5]=i-width+1;
	}
	else if(i==(height*width)-width){
		x[0]=i;
		x[1]=i+1;
		x[2]=i-width;
		x[3]=i-width+1;
	}
	else if((i>width-1 &&i<(height*width)-(2*width+1))&&i % width ==0){
		x[0]=i;
		x[1]=i+1;
		x[2]=i+width;
		x[3]=i+width+1;
		x[4]=i-width;
		x[5]=i-width+1;

	}
	else{
		x[0]=i;
		x[1]=i+1;
		x[2]=i-1;
		x[3]=i+width;
		x[4]=i+width+1;
		x[5]=i+width-1;
		x[6]=i-width;
		x[7]=i-width+1;
		x[8]=i-width-1;
	}
	int pixel = i*4;
	int c=0;
for (int i=0;i<sizeof(x)/sizeof(x[0]);i++){
	if(x[i]!=NULL){
		red+= ImageInput[x[i]*4];
		green+= ImageInput[x[i]*4+1];
		blue+= ImageInput[x[i]*4+2];
		c++;
		}

	}
		red=red/c;
		green=green/c;
		blue=blue/c;
		transperency=ImageInput[i*4+3];
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
    for (int i=0; i<totalValues; i++) {
        hostImageInput[i]  = img[i];
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
	lodepng_encode32_file("output.png", hostImageOutput, width, height);


    // free dynamic allocation
    free(img);
    free(hostImageInput);
    free(hostImageOutput);
    cudaFree(deviceImageInput);
    cudaFree(deviceImageOutput);

    return 0;
}