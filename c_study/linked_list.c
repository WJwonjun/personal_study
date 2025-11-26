#include<stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct _node{
    void *data;
    struct _node* next;
}Node;

int compareNode(Node *n1,Node *n2){
    return strcmp(n1->data,n2->data);
}

typedef void(*DISPLAY)(void *);
void displayNode(Node *node){
    printf("%s",node->data);
}
typedef int(*COMPARE)(void*,void*);
typedef struct _linkedList{
    Node *head;
    Node *tail;
    Node *current;
}LinkedList;

void initializeList(LinkedList * list){
    list->head= NULL;
    list->tail=NULL;
    list->current=NULL;
}

void addHead(LinkedList* list, void* data){
    Node *newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    if (list->head==NULL){
        list->tail = newNode;
    }
    else{
    newNode->next = list->head;
    }
    list->head = newNode;
}

void add(LinkedList* list, void* data){
    Node *newNode = (Node*)malloc(sizeof(Node));
    newNode->data = data;
    if (list->tail==NULL){
        list->head = newNode;
    }
    else{
    newNode->next = list->tail;
    }
    list->tail = newNode;
}

Node *getNode(LinkedList* list, COMPARE compare, void* data){
    Node *node = list->head;
    while (node!=NULL){
        if (compare(node->data, data)==0){
            return node;
        }
        node = node->next;
    }
    return NULL;
}

void delete(LinkedList* list, Node *node){
    if (node==list->head){
        if (list->head->next==NULL){
            list->head = list->tail=NULL;
        }
        else{
            list->head = list->head->next;
        }
    }else{
        Node *tmp = list->head;
        while (tmp!=NULL &&tmp->next!=node){
            tmp = tmp->next;
        }
        if (tmp!=NULL){
            tmp->next= node->next;

        
        if (list->tail==node){
            list->tail = tmp;
        }
    }
    }
    free(node);
}

