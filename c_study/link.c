#include<stdlib.h>
#include "link.h"

typedef struct _node
{
    Data* data;
    struct _node* next;
}Node;

struct _linkedList{
    Node* head;
};

LinkedList* getLinkedListInstance(){
    LinkedList* list = (LinkedList*)malloc(sizeof(LinkedList));
    list->head = NULL;
    return list;
}

void removeLinkedListInstance(LinkedList* list){
    Node *tmp = list->head;
    while (tmp!=NULL){
        free(tmp->data);
        Node* current = tmp;
        tmp = tmp->next;
        free(current);
    }
    free(list);
}
