#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"
#include <cuda_runtime_api.h>



//compile with c++ lodepng file
//nvcc boxBlur.cu lodepng.cpp




int main () {
    // png decode variables
    unsigned char *img;
    unsigned int height, width;

    // lodepng decode
    lodepng_decode32_file(&img, &width, &height, "hck.png");

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
    // allocate memory in gpu
    cudaMemcpy(deviceImageInput, hostImageInput, totalSpace, cudaMemcpyHostToDevice);



    return 0;
}