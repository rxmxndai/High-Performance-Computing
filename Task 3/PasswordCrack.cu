#include <stdio.h>
#include <stdlib.h>

//__global__ --> GPU function which can be launched by many blocks and threads
//__device__ --> GPU function or variables
//__host__ --> CPU function or variables

__device__ char* CudaCrypt(char* rawPassword){

	char * newPassword = (char *) malloc(sizeof(char) * 11);
	//ab12-->cdqwer35

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

//ab12
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

__device__ char  convertCharacter(char letter){
			if(letter > 122){
				letter = (letter- 122) + 97;
			}else if(letter < 97){
				letter = (97 - letter) + 97;
			}
			return letter;
}
__device__ char  convertNumber(char letter){
			if(letter > 57){
				letter = (letter- 57) + 48;
			}else if(letter < 48){
				letter = (48 - letter) + 48;
			}
			return letter;
}
__device__ char  findFirstLetter(char * letters){
	for (int i = 0; i < 3; i++)
	{
		if(i == 0){
			char possible = letters[0]-2;
			if(convertCharacter(possible - 2) == letters[1] && convertCharacter(possible+1) == letters[2]) return possible;
		}
		else if(i==1){
			char possible = letters[1]+2;
			if(convertCharacter(possible + 2) == letters[0] && convertCharacter(possible+1) == letters[2]) return possible;
		}	
		else {
			char possible = letters[2]-1;
			if(convertCharacter(possible + 2) == letters[0] && convertCharacter(possible-2) == letters[1]) return possible;
		}
	}
}

__device__ char  findSecondLetter(char * letters){
	for (int i = 0; i < 3; i++)
	{
		if(i == 0){
			char possible = letters[0]-3;
			if(convertCharacter(possible - 3) == letters[1] && convertCharacter(possible-1) == letters[2]) return possible;
		}
		else if(i==1){
			char possible = letters[1]+3;
			if(convertCharacter(possible + 3) == letters[0] && convertCharacter(possible-1) == letters[2]) return possible;
		}	
		else {
			char possible = letters[2]+1;
			if(convertCharacter(possible + 3) == letters[0] && convertCharacter(possible-3) == letters[1]) return possible;
		}
	}
}

__device__ char  findFirstDigit(char * letters){
	for (int i = 0; i < 2; i++)
	{
		if(i == 0){
			char possible = letters[0]-2;
			if(convertNumber(possible - 2) == letters[1]) return possible;
		}
		else {
			char possible = letters[1]+2;
			if(convertNumber(possible + 2) == letters[0]) return possible;
		}	
	
	}
}

__device__ char  findSecondDigit(char * letters){
	for (int i = 0; i < 2; i++)
	{
		if(i == 0){
			char possible = letters[0]-4;
			if(convertNumber(possible - 4) == letters[1]) return possible;
		}
		else {
			char possible = letters[1]+4;
			if(convertNumber(possible + 4) == letters[0]) return possible;
		}	
	
	}
}


__device__ char * CudaDecrypt (char * encryptedPassword){
	char * decryptedPassword = (char *) malloc(sizeof(char) * 5);

	// for first letter decryption;
	char foundFirstLetter;
	char firstThreeLetters []= {encryptedPassword[0] , encryptedPassword[1] , encryptedPassword[2]};
	char secondThreeLetters []= {encryptedPassword[3] , encryptedPassword[4] , encryptedPassword[5]};
	char firstTwoDigits[]= {encryptedPassword[6] , encryptedPassword[7]};
	char secondTwoDigits[]= {encryptedPassword[8] , encryptedPassword[9]};
	char test [] = {findFirstLetter(firstThreeLetters),findSecondLetter(secondThreeLetters),findFirstDigit(firstTwoDigits),
	findSecondDigit(secondTwoDigits) , '\0'};
	printf("decrypted passwoed of %s : %s \n",encryptedPassword,test);
	
} 

__global__ void crack(char * alphabet, char * numbers){

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


if(genRawPass[0] == 'a' && genRawPass[1] == 'z' && genRawPass[2] == '0' && genRawPass[3]== '2'){
printf("%c %c %c %c = %s\n", genRawPass[0],genRawPass[1],genRawPass[2],genRawPass[3], CudaCrypt(genRawPass));

char * test = CudaCrypt(genRawPass);
CudaDecrypt(test);
}


}

int main(int argc, char ** argv){

char cpuAlphabet[26] = {'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'};
char cpuNumbers[10] = {'0','1','2','3','4','5','6','7','8','9'};

char * gpuAlphabet;
cudaMalloc( (void**) &gpuAlphabet, sizeof(char) * 26); 
cudaMemcpy(gpuAlphabet, cpuAlphabet, sizeof(char) * 26, cudaMemcpyHostToDevice);

char * gpuNumbers;
cudaMalloc( (void**) &gpuNumbers, sizeof(char) * 26); 
cudaMemcpy(gpuNumbers, cpuNumbers, sizeof(char) * 26, cudaMemcpyHostToDevice);

crack<<< dim3(26,26,1), dim3(10,10,1) >>>( gpuAlphabet, gpuNumbers );
cudaThreadSynchronize();
return 0;
}













