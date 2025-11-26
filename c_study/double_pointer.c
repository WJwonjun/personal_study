#include <stdio.h>
#include <stdlib.h>

void  allocateArray(int **arr, int size, int value){
    *arr = (int*)malloc(sizeof(int)*size);
    if (*arr!=NULL){
        for (int i=0;i<size;i++){
            *(*arr+i) = value;
        }
    }
}

void safeFree(void **pp){
    if(pp!=NULL && *pp!=NULL){
        free(*pp);
        *pp = NULL;
    }
}
#define safeFree(p) safeFree((void *)&(p));
int main(){
    /*
    char **bestBooks[3];
    char **englishBooks[4];

    char *titles[] = {"A","B","C","D","E","F"};
    
    bestBooks[0] = &titles[0];
    bestBooks[1] = &titles[3];
    bestBooks[2] = &titles[5];

    englishBooks[0] = &titles[0];
    englishBooks[1] = &titles[1];
    englishBooks[2] = &titles[2];
    englishBooks[3] = &titles[4];

    printf("%s",*englishBooks[0]);
    */

    int *vector  = NULL;
    allocateArray(&vector,5,45);
    printf("%d",vector[0]);


}