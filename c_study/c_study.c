#include<stdio.h>
#include<stdlib.h>

int** create_matrix(int,int);
void destroy_matrix(int**,int);

int main(){
    int row, col,i,j,**matrix;
    printf("Enter row and col.");
    scanf("%d%d",&row,&col);

    matrix=create_matrix(row,col);
    for (i=0;i<row;i++){
        for (j=0;j<col;j++){
            matrix[i][j] = i*i+j*j;
            printf("%d\t",matrix[i][j]);
        }
        printf("\n");
    }
    destroy_matrix(matrix,row);
    return 0;
}

int** create_matrix(int row, int col){
    int i, **p;
    p = (int**)malloc(row*sizeof(int*));
    if (p==NULL) exit(1);
    for (i=0;i<row;i++){
        p[i] = (int*)malloc(col*sizeof(int));
        if (p[i]==NULL)exit(1);
    }
    return p;
}

void destroy_matrix(int** p,int row){
    int i;
    for (i=0;i<row;i++)
        free(p[i]);
    free(p);
}