#include <stdio.h>
#include <string.h>
#include <stdlib.h>

char* trim(char* phrase){
    char* old = phrase;
    char* new = phrase;

    while (*old==' '){
        old++;
    }
    while (*old){
        *(new++)=*(old++);
    }
    *new=0;
    return (char*) realloc(phrase, strlen(phrase)+1);
}

int main(){
    char* word = (char*)malloc(strlen(" cat")+1);
    strcpy(word," cat");
    printf("%s\n",trim(word));
}