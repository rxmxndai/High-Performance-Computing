#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>



// to get the start and end value of the iterative loop for each thread
struct threadInfo
{
    int start;
    int end;
};



// stores a matrix's row / dimension / pointer to that matrix
typedef struct {
    int row;
    int column;
    double **pMat;
} matrix;



// Overall argument to be passed with threads
typedef struct args_struct
{
    int matC_elements;
    int rows_matA;
    int cols_matA;
    int rows_matB;
    int cols_matB;
    double **matrixA_ptr;
    double **matrixB_ptr;
    double **output_matrix_ptr;
    // int thread_count;
    struct threadInfo threadDetails[100];

    int start;
    int end;
} __args;




int NUM_THREADS;
pthread_mutex_t mutex;
void *multiplyMatrices(void *param);

FILE *fptrA, *fptrB, *fptrC;
matrix matA, matB, matC;



// *******************************************************************************************************************************88 //
void checkArguments(int argc, char **argv) {
    if (argc <= 1 || argc >= 3) {
        printf("Error passing arguments!");
        exit(1);
    }
    NUM_THREADS = strtol(argv[1], NULL, 10);

    if (NUM_THREADS >= 1000 || NUM_THREADS < 1) {
        printf("Error passing arguments!");
        exit(1);
    }
}




// ******************************************************************************************************************************* //

// function to make a new 2d array that receives a pointer and 
// returns new 2d array with 0 as default value for all index
double ** initializeNew2Darray(int *dimension) {
    double **arr;

    int r = dimension[0], c = dimension[1];

    arr = (double **) malloc(sizeof(double *) * r); 
    for (int i=0; i<r; i++) {
        arr[i] =  (double *) malloc(sizeof(double) * c);
        for(int j = 0; j < c; j++) {
            arr[i][j] = 0;
        }
    }

    return arr;
}



// ******************************************************************************************************************************* //
// Receives a matrix (struct) that contains rows, columns and the actual 2d arrray of image file
void displayMatrix(const matrix m) {
    int r=m.row, c=m.column;

    double **arr;
    arr = m.pMat;
    for (int i=0; i<r; i++ ) {
        for (int j=0; j<c; j++) {
            printf("%.2lf  ", *(*(arr+i)+j) );
            
        }
        printf("\n");
    }
}




// ******************************************************************************************************************************* //
// receives a filename and reads through each character in the file,
// returns number of rows and columns of that file data
int * getDimension(const char *filename) {
    FILE *fptr;
    fptr = fopen(filename, "r");
    char ch;
    int rows=1, columns=1;

    int *dimension = (int *) malloc (sizeof(int) * 2);

    while ( 1 ) {
     ch = fgetc ( fptr ) ; // reading the file
     
     if (ch == '\n') {
        columns = 1;
        rows++;
     }
     else if (ch == ',') {
        columns++;
     }
     else if ( ch == EOF ) {
        break ;
     }
   }

    dimension[0] = rows;
    dimension[1] = columns;

    printf("%d\t%d\n", dimension[0], dimension[1]);
    fclose(fptr);
    return dimension;
}



// *******************************************************************************************************************************//
// checks weather the number of thread exceeds the total number of elements in eiter matrices

void canMultiply(int rowA, int rowB, int colA, int colB) {
    int dimensionA = rowA * colA;
    int dimensionB = rowB * colB;

    if (NUM_THREADS > dimensionA || NUM_THREADS > dimensionB) {
        if (dimensionA > dimensionB) {
            NUM_THREADS = dimensionA;
        }
        else {
            NUM_THREADS = dimensionB;
        }
        printf("\nNUM_THREADS capped to %d\n", NUM_THREADS);
    }
}




// ******************************************************************************************************************************* //
// receives the array that needs to be stored with files' values, the filename from where to store, 
// and the dimension of the matrix where rows and columns are specified
void storeMatrixFromFile(double **arr, const char *filename, int *dimension) {
    int row = *(dimension+0), column = *(dimension+1);
    FILE *fptr = fopen(filename, "r");

    for (int i=0; i<row; i++) {
        for (int j=0; j<column; j++) {
             fscanf(fptr, "%lf,", &arr[i][j]);
        }
    }
}





// *******************************************************************************************************************************88 //
void writeMatrixToFile(matrix mat, int * d) {

    // int row=*(d+0), column=*(d+1);

    int row = mat.row;
    int column = mat.column;

    double **arr = mat.pMat;


    printf("\nFinal Matrix: Row %d\t Column: %d\n\n", row, column);


    for (int i=0; i<row; i++) {
        for (int j=0; j<column; j++) {
            if (j == column - 1) {
                fprintf(fptrC, "%lf", *(*(arr+i) + j));
            }
            else {
                fprintf(fptrC, "%lf,", *(*(arr+i) + j));
            }
        }
        fprintf(fptrC, "\n");
    }

}




// ************************* Entry point of the CODE *********************************************************88 //

void main (int argc, char **argv) {

    // check for correct num threads passed through CLI and assigns NUMTHREAD
    checkArguments(argc, argv);

    printf("NUM_THREADS %d\n", NUM_THREADS);

    char *matrixA_fileName = "Mat1.txt";
    char *matrixB_fileName = "Mat2.txt";
    char *resultantMatrix_filename = "MultipliedMatrix.txt";
    
    // FOPEN three files
    fptrA = fopen(matrixA_fileName, "r");
    fptrB = fopen(matrixB_fileName, "r");
    fptrC = fopen(resultantMatrix_filename, "w");



    // dimension stores rows and columns of respective matrix
    int *dimensionA, *dimensionB, *dimensionC;
    int rowA, rowB, rowC, colA, colB, colC;

    dimensionA = getDimension(matrixA_fileName);
    dimensionB = getDimension(matrixB_fileName);
    rowA = *(dimensionA + 0);
    colA = *(dimensionA + 1);
    
    rowB = *(dimensionB + 0);
    colB = *(dimensionB + 1);

    
    printf("\nMatrix A\t\t Rows: %d\t Columns: %d\n", rowA, colA);
    printf("Matrix B\t\t Rows: %d\t Columns: %d\n\n", rowB, colB);


    rowC = rowA;
    colC = colB;
    
    // weather two matrices are multiplyable by threads
    /// thread limitter
    canMultiply(rowA, rowB, colA, colB);

    if (rowB != colA) {
        printf("The following matrices cannot be multiplied !\n");
        exit(0);
    }


    
    // ****************************************************************** //
    
    // Make a new dynamic 2d array with double pointers
    double **arrA, **arrB, **arrC;
    arrA = initializeNew2Darray(dimensionA);
    storeMatrixFromFile(arrA, matrixA_fileName, dimensionA);

    arrB = initializeNew2Darray(dimensionB);
    storeMatrixFromFile(arrB, matrixB_fileName, dimensionB);

    printf("\n ******************** \n");
    dimensionC[0] = rowC;
    dimensionC[1] = colC;
    arrC = initializeNew2Darray(dimensionC);
    

    matA.row = rowA;
    matA.column = colA;
    matA.pMat = arrA;

    matB.row = rowB;
    matB.column = colB;
    matB.pMat = arrB;

    matC.row = rowC;
    matC.column = colC;
    matC.pMat = arrC;

    printf("\n ********** Matrix A elements ********** \n");
    displayMatrix(matA);
    printf("\n ********** Matrix B elements ********** \n");
    displayMatrix(matB);
   

     // ******************************************************************* //

    long singlethreadtask = rowC / NUM_THREADS;
    long remainingiteration =  rowC % NUM_THREADS;

    // make an array of total threads count
    // tbp array is threads breakpoints
    long *tbp = (long *) malloc (NUM_THREADS * sizeof (long));


    for (int i = 0; i < NUM_THREADS; i++) {
        tbp[i] = singlethreadtask;      
    }  


    for (int i = 0; i < remainingiteration; i++) {
        tbp[i]++;
    }


    // threads that is to be passed can be managed through these starting and ending array
    long starting[NUM_THREADS], ending[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; i++ ) {
        if (i == 0) {
            starting[i] = 0;
        }
        else {
             starting[i] = ending[i-1] + 1;
        }
        ending[i] = starting[i] + tbp[i] - 1;
    }

     // **************************************************************************** //
    //Initialize threads 
     pthread_t thread_id[NUM_THREADS];
    // Initialize thread arguments
    __args threadArgs[NUM_THREADS];


    // global data for all threads
     for (int i=0; i<NUM_THREADS; i++) {
            threadArgs[i].cols_matA = colA;
            threadArgs[i].rows_matA = rowA;
            threadArgs[i].cols_matB = colB;
            threadArgs[i].rows_matB = rowB;

            threadArgs[i].matC_elements = rowC*colC;
            
            threadArgs[i].matrixA_ptr = arrA;
            threadArgs[i].matrixB_ptr = arrB;
            threadArgs[i].output_matrix_ptr = arrC;
     }
            


    // **************************************************************************** //
    
    pthread_mutex_init (&mutex, NULL);


    // invoke thread function by all threads with start and end limit for all threads in threadArgs
        for (int m = 0; m < NUM_THREADS; m++) {
            threadArgs[m].start = starting[m];
            threadArgs[m].end = ending[m];
            pthread_create(&thread_id[m], NULL, &multiplyMatrices, (void *)&threadArgs[m]);
        }
        for (int n = 0; n < NUM_THREADS; n++) {
            pthread_join(thread_id[n], NULL);
        }
     
    printf("\n ********** Matrix C elements ********** \n");
    displayMatrix(matC);

    dimensionC[0] = rowC;
    dimensionC[1] = colC;
    
    // writing to txt file sending matrix
    writeMatrixToFile(matC, dimensionC);
    printf("\n ******************** \n");
    printf("\nOutputMatrix written in file '%s'\n", resultantMatrix_filename);
    printf("\n ******************** \n");

    pthread_mutex_destroy(&mutex);
    free(arrA);
    free(arrB);
    free(arrC);
}





// *******************************************************************************************************************************88 //


// thread function to process the number of rows assigned to each thread
void *multiplyMatrices(void *args)
{
    
    // getting thread arguments data
    struct args_struct *arguments = args;

    int colA = arguments->cols_matA;
    int rowA = arguments->rows_matA;
    int colB = arguments->cols_matB;
    int rowB = arguments->rows_matB;

    int startLimit = arguments->start;
    int endLimit = arguments->end;

    double **arrA = arguments->matrixA_ptr;
    double **arrB = arguments->matrixB_ptr;
    double **arrC = arguments->output_matrix_ptr;

    int colC = rowA;


    printf("StartLimit: %d,\t EnLimit: %d\n", startLimit, endLimit);

    ///////////////////////////////////////////////////////////////////////////////////
    pthread_mutex_lock(&mutex);
    double sum = 0.0;
    
    for (int i = startLimit; i <= endLimit; i++)
    {   
        // calcualte from left to right of final matrix
        for (int j = 0; j < colC; j++)
        {
            // calculate from top to bottom of final matrix
            for (int k = 0; k < rowB; k++)
            {
                sum += *(*(arrA+i)+k) * *(*(arrB+k)+j);
            }

            *(*(arrC+i) +j) = sum;
            sum = 0.0;
        }
        printf("\n");
    }

    pthread_mutex_unlock(&mutex);
}


