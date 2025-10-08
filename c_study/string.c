#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

char* stringToLower(const char* string){
    char *tmp = (char*)malloc(strlen(string)+1);
    char *start = tmp;
    while (*string!=0){
        *tmp++ = tolower(*string++);
    }
    *tmp=0;
    return start;
}

int compareIgnorecase(const char* s1, const char* s2){
    char* t1 = stringToLower(s1);
    char* t2 = stringToLower(s2);

    int result = strcmp(t1,t2);
    free(t1);
    free(t2);
    return result;
}
typedef int (*fptrOperation)(const char*,const char*);

void sort(char *array[],int size, fptrOperation operation){
    int swap=1;
    while (swap){
        swap=0;
        for(int i=0;i<size-1;i++){
            if(operation(array[i],array[i+1])>0){
                swap=1;
                char *tmp = array[i];
                array[i] = array[i+1];
                array[i+1] = tmp;
            }
        }
    }
}

void displayNames(char *names[], int size){
    for (int i=0;i<size;i++){
        printf("%s ",names[i]);
    }
    printf("\n");
}

int main(){
    char* names[] = {"BOB","Ted","Carol","Alice","alice"};
    sort(names,5,compareIgnorecase );
    displayNames(names,5);
}