#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime_api.h>



//compile with c++ lodepng file
//nvcc boxBlur.cu lodepng.cpp

__global__ void boxBlur(unsigned char *Image, unsigned char * ImageOuput, int b, int a){

	int x[]={NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL};

	int red=0,green=0,blue=0,trans=0;
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i==0){
		x[0]=i;
		x[1]=i+1;
		x[2]=i+b;
		x[3]=i+b+1;

	}
	else if(i>0 && i<(b-1)){
		x[0]=i;
		x[1]=i+1;
		x[2]=i-1;
		x[3]=i+b;
		x[4]=1+i+b;
		x[5]=i+b-1;
	}
	else if (i==(b-1)){
		x[0]=i;
		x[1]=i-1;
		x[2]=i+b;
		x[3]=i+b-1;
	}
	else if(((i > b-1 && i< (a*b)-b) && ((i+1) % b ==0))){
		x[0]=i;
		x[1]=i-1;
		x[2]=i-b;
		x[3]=i-b-1;
		x[4]=i+b;
		x[5]=i+b-1;
	}
	else if (i==((a*b)-1)){
		x[0]=i;
		x[1]=i-1;
		x[2]=i-b-1;
		x[3]=i-b;
	}
	else if(i>((a*b)-b) && i < (a*b)){
		x[0]=i;
		x[1]=i+1;
		x[2]=i-1;
		x[3]=i-b;
		x[4]=i-b-1;
		x[5]=i-b+1;
	}
	else if(i==(a*b)-b){
		x[0]=i;
		x[1]=i+1;
		x[2]=i-b;
		x[3]=i-b+1;
	}
	else if((i>b-1 &&i<(a*b)-(2*b+1))&&i % b ==0){
		x[0]=i;
		x[1]=i+1;
		x[2]=i+b;
		x[3]=i+b+1;
		x[4]=i-b;
		x[5]=i-b+1;

	}
	else{
		x[0]=i;
		x[1]=i+1;
		x[2]=i-1;
		x[3]=i+b;
		x[4]=i+b+1;
		x[5]=i+b-1;
		x[6]=i-b;
		x[7]=i-b+1;
		x[8]=i-b-1;
	}
	int pixel = i*4;
	int c=0;
for (int i=0;i<sizeof(x)/sizeof(x[0]);i++){
	if(x[i]!=NULL){
		red+= Image[x[i]*4];
		green+= Image[x[i]*4+1];
		blue+= Image[x[i]*4+2];
		c++;
		}

	}
		red=red/c;
		green=green/c;
		blue=blue/c;
		trans=Image[i*4+3];
		ImageOuput[pixel] = red;
		ImageOuput[1+pixel] = green;
		ImageOuput[2+pixel] = blue;
		ImageOuput[3+pixel] = trans;
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