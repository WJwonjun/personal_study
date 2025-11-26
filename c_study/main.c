#include "point.h"
#include <stdlib.h>
#include <stdio.h>

int main(){
    struct rectangle *p;
    int x_diff,y_diff;

    p = (struct rectangle *)malloc(sizeof(struct rectangle));
    if (p==NULL){
        printf("No more memory.\n");
        exit(1);
    }

    printf("Enter x,y of top left.\n");
    scanf("%d%d",&(p->tl).x,&(p->tl).y);

    printf("Enter x,y, of bottom right.\n");
    scanf("%d%d",&(p->br).x,&(p->br).y);

    x_diff = abs((p->br).x-(p->tl).x);
    y_diff = abs((p->br).y-(p->tl).y);
    printf("The area of rectangle is, %d.\n",x_diff*y_diff);
    return 0;
}