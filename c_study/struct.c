#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#define LIST_SIZE 10

typedef struct _person{
    char* firstname;
    char* lastname;
    char* title;
    unsigned int age;
}Person ;
Person *list[LIST_SIZE];
/*
int main(){
    Person *ptrPerson;
    ptrPerson = (Person *)malloc(sizeof(Person));
    ptrPerson->firtstname = (char *)malloc(strlen("Emily")+1);
    strcpy(ptrPerson->firtstname,"Emily");
    ptrPerson->age = 23;
}
*/
void initialization(){
    for (int i=0;i<LIST_SIZE;i++){
        list[i] = NULL;
    }
}
void initializePerson(Person *person,const char*fn, const char* ln, const char* title, unsigned int age){
    person->firstname = (char*)malloc(strlen(fn)+1);
    strcpy(person->firstname,fn);
    person->lastname = (char*)malloc(strlen(ln)+1);
    strcpy(person->lastname,ln);
    person->title = (char*)malloc(strlen(title)+1);
    strcpy(person->title, title);
    person->age = age;
}

Person *getPerson(){
    for(int i=0;i<LIST_SIZE;i++){
        if (list[i]!=NULL){
            Person *ptr = list[i];
            list[i] = NULL;
            return ptr;
        }
    }
    Person *person = (Person*)malloc(sizeof(Person));
    return person;
}
void deallocatePerson(Person *person){
    free(person->firstname);
    free(person->lastname);
    free(person->title);
}
Person *returnPerson(Person *person){
    for (int i=0;i<LIST_SIZE;i++){
        if(list[i]==NULL){
            list[i]=person;
            return person;
        }
    }
    deallocatePerson(person);
    free(person);
    return NULL;
}
void displayPerson(Person *person){
    printf("%s %s %s %d",person->firstname,person->lastname,person->title,person->age);
}


int main(){
    initialization();
    Person *ptrPerson;
    ptrPerson = getPerson();

    initializePerson(ptrPerson,"Ralph","Fitsgerald","Mr.",35);
    displayPerson(ptrPerson);
    returnPerson(ptrPerson);
}