#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>


// Guide: 
// Compile with: gcc MatrixMultiplicationMultithreads.c -pthread -o a -lpthread
// Run with: ./a (number of threads)

int threadCount;

// structure to store the individual martix elements
typedef struct
{
    double x;
} unit;

// structure to store the individual matrix rows, columns
// and the reference to the matrix elements
typedef struct
{
    int rows;
    int cols;
    unit *x;
} matrix;

matrix A, B, C, _C, target;

pthread_mutex_t mutex;

// function prototypes
void errorThreads(char *argument);
void selectThreadsNum(int argc, char *argv[]);
void checkAvaibility(int rows_matA, int cols_matA, int rows_matB, int cols_matB);
int *countRowsAndColumns(const char *file_name);
matrix newMatrix(int rows, int cols);
void displayMatrix(matrix displayable_matrix);
double calculateOneIteration(int first, int second);
void *multiplyMatrices(void *args);

// the main function that invokes itself at runtime, obviously!
void main(int argc, char *argv[])
{
    selectThreadsNum(argc, argv);

    FILE *fptr1, *fptr2, *fptr3 = NULL;
    int row, col;

    int rowsA, columnA, rowsB, columnB, rowsC, columnC;

    char *matrixA_filename = "Mat1.txt";
    char *matrixB_filename = "Mat2.txt";
    char *output_matrix_filename = "outputMatrix.txt";

    fptr1 = fopen(matrixA_filename, "r");
    fptr2 = fopen(matrixB_filename, "r");
    fptr3 = fopen(output_matrix_filename, "w");

    if (fptr1 != NULL && fptr2 != NULL && fptr3 != NULL)
    {
        int *p;
        int *q;

        p = countRowsAndColumns(matrixA_filename);

        rowsA = *(p + 0);
        columnA = *(p + 1);


        q = countRowsAndColumns(matrixB_filename);

        rowsB = *(q + 0);
        columnB = *(q + 1);

        checkAvaibility(rowsA, columnA, rowsB, columnB);

        // output matrix C is the combination of rows from matrix A and columns from matrix B
        rowsC = rowsA;
        columnC = columnB;

        printf("\nMatrix A: Rows: %d, Columns: %d\n", rowsA, columnA);
        printf("Matrix B: Rows: %d, Columns: %d\n", rowsB, columnB);
        printf("Output Matrix:  Rows: %d, Columns: %d\n\n", rowsA, columnB);

        if (columnA == rowsB)
        {
            matrix target_matA, target_matB;
            target_matA.rows = rowsA;
            target_matA.cols = columnA;

            target_matB.rows = rowsB;
            target_matB.cols = columnB;

            // dynamic memory allocation
            target_matA.x = (unit *)malloc(rowsA * columnA * sizeof(unit));
            target_matB.x = (unit *)malloc(rowsB * columnB * sizeof(unit));

            if (target_matA.x == NULL || target_matB.x == NULL)
            {
                printf("\nError! memory not allocated.\n");
                exit(1);
            }

            // Scanning the file and storing the matrix A data in allocated memory
            for (row = 0; row < rowsA; row++)
            {
                for (col = 0; col < columnA; col++)
                {
                    fscanf(fptr1, "%lf,", &(target_matA.x + row * target_matA.cols + col)->x);
                }
            }

            // printing the matrix A elements
            printf("\nMatrix A elements:\n");
            A = target_matA;
            displayMatrix(A);

            // Scanning the file and storing the matrix B data in allocated memory
            for (row = 0; row < rowsB; row++)
            {
                for (col = 0; col < columnB; col++)
                {
                    fscanf(fptr2, "%lf,", &(target_matB.x + row * target_matB.cols + col)->x);
                }
            }

            // printing the matrix B elements
            printf("Matrix B elements: \n");
            B = target_matB;
            displayMatrix(B);

            int i;
            C = newMatrix(A.rows, B.cols);
            for (i = 0; i < C.cols * C.rows; i++)
            {
                (C.x + i)->x = 0.0;
            }

            _C = newMatrix(A.rows, B.cols);
            for (i = 0; i < _C.cols * _C.rows; i++)
            {
                (_C.x + i)->x = 0.0;
            }

            pthread_t thread_id[threadCount];

            printf("Thread Task Started !\n\n");

            pthread_mutex_init(&mutex, NULL);

            for (int m = 0; m < threadCount; m++)
            {
                pthread_create(&thread_id[m], NULL, multiplyMatrices, NULL);
            }

            for (int n = 0; n < threadCount; n++)
            {
                pthread_join(thread_id[n], NULL);
            }

            // writing the output matrix C into the file `matrixresults2050423.txt`
            for (i = 0; i < rowsC; i++)
            {
                for (int j = 0; j < columnC; j++)
                {
                    if (j == columnC - 1)
                        fprintf(fptr3, "%lf", (C.x + i * columnC + j)->x);
                    else
                        fprintf(fptr3, "%lf,", (C.x + i * columnC + j)->x);
                }
                fprintf(fptr3, "\n");
            }

            printf("\nOutput matrix C elements: \n");
            displayMatrix(C);

            printf("Output matrix stored in file named: %s\n\n", output_matrix_filename);

            // deallocating the memory
            free(target_matA.x);
            free(target_matB.x);
            free(target.x);
        }
        else
        {
            printf("\nOops! the column of matrix A is not equal to the row of matrix B, thus matrices cannot be multiplied.\n");
        }

        fclose(fptr1);
        fclose(fptr2);
        fclose(fptr3);
    }
    else
    {
        printf("\nNo such file found!\n");
    }
}

// function to display a message explaining what and how arguments should be passed
void errorThreads(char *argument)
{
    fprintf(stderr, "arguments should be in the order as specified:   %s   <number of threads>\n", argument);
    fprintf(stderr, "where number of threads should be > 0 and < 1000\n");
    exit(1);
}

// function to get the command line arguments
void selectThreadsNum(int argc, char *argv[])
{
    if (argc != 2)
    {
        errorThreads(argv[0]);
    }

    threadCount = strtol(argv[1], NULL, 10);

    if (threadCount <= 0 || threadCount >= 1000)
    {
        errorThreads(argv[0]);
    }
}

// function to handle and limit the number of input threads to the dimensions of the matrices
void checkAvaibility(int rows_matA, int cols_matA, int rows_matB, int cols_matB)
{
    if (threadCount > rows_matA * cols_matA || threadCount > rows_matB * cols_matB)
    {
        // limiting the MAX_THREADS_ALLOWED to the maximum dimension of matrices
        if (rows_matA * cols_matA > rows_matB * cols_matB)
            threadCount = rows_matA * cols_matA;
        else
            threadCount = rows_matB * cols_matB;
        printf("\nMAX_THREADS_ALLOWED: %d\n", threadCount);
        // printf("since input number of threads should not be greater than the number of rows or columns to be processed\n");
    }
}

// function to find the number of rows and columns of each matrix from the files
int *countRowsAndColumns(const char *file_name)
{
    FILE *fp = fopen(file_name, "r");
    int newRows = 1;
    int newCols = 1;
    char ch;

    static int rows_cols[10];

    while (!feof(fp))
    {
        ch = fgetc(fp);

        if (ch == '\n')
        {
            newRows++;
            // rows_cols[0] = newCols;
            newCols = 1;
        }
        else if (ch == ',')
        {
            newCols++;
        }
    }
    rows_cols[0] = newRows;
    rows_cols[1] = newCols;


    return rows_cols;
}

matrix newMatrix(int rows, int cols)
{
    // matrix target;
    int i, j;
    double temp_data;

    target.rows = rows;
    target.cols = cols;
    target.x = (unit *)malloc(rows * cols * sizeof(unit));
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
        {
            temp_data = 0.0;
            (target.x + i * target.cols + j)->x = temp_data;
        }
    return target;
}

void displayMatrix(matrix displayable_matrix)
{
    int rows = displayable_matrix.rows;
    int cols = displayable_matrix.cols;
    int i, j;

    for (i = 0; i < rows; i++)
    {
        printf("[  ");
        for (j = 0; j < cols; j++)
            printf("%f  ", (displayable_matrix.x + i * cols + j)->x);
        printf("]\n");
    }
    printf("\n\n");
}

// function to return a multiplied matrix unit
double calculateOneIteration(int first, int second)
{
    int i;
    double res = 0.0;

    for (i = 0; i < A.cols; i++)
    {
        res += (A.x + first * A.cols + i)->x * (B.x + i * B.cols + second)->x;
    }

    return res;
}

// thread function to process the number of rows assigned to each thread
void *multiplyMatrices(void *param)
{
    while (1)
    {
        int firstNum;
        int secondNum;
        int i, j, flag = 0, close = 0;
        double res;

        pthread_mutex_lock(&mutex);
        for (i = 0; i < _C.rows; i++)
        {
            for (j = 0; j < _C.cols; j++)
            {
                if ((_C.x + i * _C.cols + j)->x == 0.0)
                {
                    firstNum = i;
                    secondNum = j;
                    (_C.x + i * _C.cols + j)->x = 1.0;
                    close = 1;
                    break;
                }
            }
            if (close == 1)
                break;
            else if (i == _C.rows - 1)
                flag = 1;
        }
        pthread_mutex_unlock(&mutex);

        if (flag == 1)
            pthread_exit(NULL);
        res = calculateOneIteration(firstNum, secondNum);
        (C.x + firstNum * C.cols + secondNum)->x = res;
    }
    pthread_exit(NULL);
}