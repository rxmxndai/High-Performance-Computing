#include <stdio.h>
#include <stdlib.h>


int main () {
    FILE *fptr;
    fptr = fopen("matrixa.txt", "r");
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

    printf("\n%d\t%d\n", dimension[0], dimension[1]);
    fclose(fptr);

    return 0;
}