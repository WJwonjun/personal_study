#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(){
int rows=2;
int columns=5;

int **matrix = (int **)malloc(rows * sizeof(int *));
matrix[0] = (int *)malloc(rows*columns*sizeof(int));
for (int i=1;i<rows;i++)
    matrix[i] = matrix[0]+ i*columns;
}