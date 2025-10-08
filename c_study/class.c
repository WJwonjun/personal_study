#include<stdio.h>
#include<stdlib.h>


typedef void (*fptrSet)(void*,int);
typedef int (*fptrGet)(void*);
typedef void (*fptrDisplay)();

typedef struct _fuctions{
    fptrSet setx;
    fptrGet getx;
    fptrSet sety;
    fptrGet gety;
    fptrDisplay display;
}vFunctions;


typedef struct _shape{
    vFunctions functions;
    int x;
    int y;
}Shape;

typedef struct _rectangle
{
    Shape base;
    int width;
    int height;
}Rectangle;

void shapeDisplay(){printf("shape");}
void shapesetx(void *shape,int x){((Shape*)shape)->x = x;}
int shapegetx(void *shape){return ((Shape*)shape)->x;}
void shapesety(void *shape,int y){((Shape*)shape)->y = y;}
int shapegety(void *shape){return ((Shape*)shape)->y;}

void recshapeDisplay(){printf("rectangle");}
void recshapesetx(void *shape,int x){((Rectangle*)shape)->base.x = x;}
int recshapegetx(void *shape){return ((Rectangle*)shape)->base.x;}
void recshapesety(void *shape,int y){((Rectangle*)shape)->base.y = y;}
int recshapegety(void *shape){return ((Rectangle*)shape)->base.y;}

Shape* getShapeInstance(){
    Shape* shape = (Shape*)malloc(sizeof(Shape));
    shape->functions.display = shapeDisplay;
    shape->functions.setx = shapesetx;
    shape->functions.getx = shapegetx;
    shape->functions.sety = shapesety;
    shape->functions.gety = shapegety;
    shape->x = 100;
    shape->y = 100;
    return shape;
}



Rectangle* getRectangleInstance(){
    Rectangle *rectangle = (Rectangle*)malloc(sizeof(Rectangle));
    rectangle->base.functions.display = recshapeDisplay;
    rectangle->base.functions.setx = recshapesetx;
    rectangle->base.functions.getx = recshapegetx;
    rectangle->base.functions.sety = recshapesety;
    rectangle->base.functions.gety = recshapegety;
    rectangle->base.x = 100;
    rectangle->base.y = 100;
    rectangle->height = 300;
    rectangle->width = 500;
    return rectangle;
}

int main(){
Rectangle *rptr = getRectangleInstance();
rptr->base.functions.setx(rptr,35);
rptr->base.functions.display();
printf(" %d",rptr->base.functions.getx(rptr));
}