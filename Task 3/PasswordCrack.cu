#include <stdio.h>
#include <stdlib.h>

//__global__ --> GPU function which can be launched by many blocks and threads
//__device__ --> GPU function or variables
//__host__ --> CPU function or variables

// Compile with: nvcc password_cracking_using_cuda.cu -o password_cracking_using_cuda
// Execute with: ./password_cracking_using_cuda


// cuda function to copy string from one variable into another since 
// strcpy(), a C function cannot be used in GPU/device
__device__ char * copy_strings(char *dest, const char *src){
  int i = 0;
  do {
    dest[i] = src[i];}
  while (src[i++] != 0);
  return dest;
}


// cuda function to compare two character arrays since strcmp(), 
// a yet another C function that cannot be used in GPU/device
__device__ int compare_strings(const char *str_a, const char *str_b, unsigned len = 256){
	int match = 0;
	unsigned i = 0;
	unsigned done = 0;
	while ((i < len) && (match == 0) && !done) {
		if ((str_a[i] == 0) || (str_b[i] == 0)) {
			done = 1;
		}
		else if (str_a[i] != str_b[i]) {
			match = i+1;
			if (((int)str_a[i] - (int)str_b[i]) < 0) {
				match = 0 - (i + 1);
			}
		}
		i++;
	}
	return match;
  }


__device__ char* CudaCrypt(char* rawPassword){

	char * newPassword = (char *) malloc(sizeof(char) * 11);
 
	newPassword[0] = rawPassword[0] + 2;
	newPassword[1] = rawPassword[0] - 2;
	newPassword[2] = rawPassword[0] + 1;
	newPassword[3] = rawPassword[1] + 3;
	newPassword[4] = rawPassword[1] - 3;
	newPassword[5] = rawPassword[1] - 1;
	newPassword[6] = rawPassword[2] + 2;
	newPassword[7] = rawPassword[2] - 2;
	newPassword[8] = rawPassword[3] + 4;
	newPassword[9] = rawPassword[3] - 4;
	newPassword[10] = '\0';

	for(int i =0; i<10; i++){
		if(i >= 0 && i < 6){ //checking all lower case letter limits
			if(newPassword[i] > 122){
				newPassword[i] = (newPassword[i] - 122) + 97;
			}else if(newPassword[i] < 97){
				newPassword[i] = (97 - newPassword[i]) + 97;
			}
		}else{ //checking number section
			if(newPassword[i] > 57){
				newPassword[i] = (newPassword[i] - 57) + 48;
			}else if(newPassword[i] < 48){
				newPassword[i] = (48 - newPassword[i]) + 48;
			}
		}
	}
	return newPassword;
}


__global__ void crack(char * alphabet, char * numbers, char * encPassword) {

	char genRawPass[4];

	genRawPass[0] = alphabet[blockIdx.x];
	genRawPass[1] = alphabet[blockIdx.y];

	genRawPass[2] = numbers[threadIdx.x];
	genRawPass[3] = numbers[threadIdx.y];

	//firstLetter - 'a' - 'z' (26 characters)
	//secondLetter - 'a' - 'z' (26 characters)
	//firstNum - '0' - '9' (10 characters)
	//secondNum - '0' - '9' (10 characters)

	//Idx --> gives current index of the block or thread


	// compare encrypted passwords and then after the match, assign the device variable with raw password
	if (compare_strings(CudaCrypt(genRawPass), encPassword) == 0) {
		// printf("%c %c %c %c = %s\n", genRawPass[0], genRawPass[1], genRawPass[2], genRawPass[3], CudaCrypt(genRawPass));
		// printf("Encrypted Password: %s\tRaw Password: %s\n", CudaCrypt(genRawPass), genRawPass);
		// copy_strings(storeRawPass, genRawPass);
		copy_strings(encPassword, genRawPass);
		// printf("storeRawPass: %s\n", storeRawPass);
	}
}


int main(int argc, char ** argv){
	char cpuAlphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
	char cpuNumbers[26] = {'0','1','2','3','4','5','6','7','8','9'};

	// the password to be encrypted using the function `CudaCrypt` and later decrypted using the function `crack`
	// char inputPassword[26] = "hd07";

	// encrypted password for `hd07` as input 
	char inputEncPass[20] = "jfigac2223";


	// local/host variable to store the decrypted password
	char *decryptedPass;

	// allocating space for local/host copy
	decryptedPass = (char *)malloc(sizeof(char) * 26);


	// allocate memory for device variables
	char * gpuAlphabet;
	cudaMalloc( (void**) &gpuAlphabet, sizeof(char) * 26); 
	cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

	char * gpuNumbers;
	cudaMalloc( (void**) &gpuNumbers, sizeof(char) * 26); 
	cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 26, cudaMemcpyHostToDevice);

	// gpu/device memory allocation for encrypted input password
	char *gpuPassword;
	cudaMalloc( (void**) &gpuPassword, sizeof(char) * 26);
	cudaMemcpy(gpuPassword, inputEncPass, sizeof(char) * 26, cudaMemcpyHostToDevice);


	crack<<< dim3(26,26,1), dim3(10,10,1) >>>( gpuAlphabet, gpuNumbers, gpuPassword );
	cudaDeviceSynchronize();  // cudaDeviceSynchronize() and cudaThreadSynchronize() works the same


	// copy the memory back to host from device
	cudaMemcpy(decryptedPass, gpuPassword, sizeof(char) * 26, cudaMemcpyDeviceToHost);
	// cudaMemcpy(decryptedPass, storeRawPassword, sizeof(char) * 26, cudaMemcpyDeviceToHost);
	

	printf("\nEncrypted Password: %s,\tRaw Password: %s\n\n", inputEncPass, decryptedPass);


	/*
	* // optional (sort of an alternative) way of doing the similar thing with less code (less variables) obviously
	* // and thus the variable decryptedPass can be removed
	* cudaMemcpy(inputEncPass, gpuPassword, sizeof(char) * 26, cudaMemcpyDeviceToHost);
	* printf("Encrypted Password: %s,\tRaw Password: %s\n", inputEncPass, inputEncPass);
	*/


	// device memory management by reallocating the memory for reusability since 
	// cudaFree() and, free() for that matter, does not actually free the memory but 
	// clears the data inside of it to be reused later... again
	free(decryptedPass);
	cudaFree(gpuAlphabet);
	cudaFree(gpuNumbers);
	cudaFree(gpuPassword);

	return 0;
}