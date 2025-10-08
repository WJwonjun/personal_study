#include <stdio.h>

int add(int num1, int num2){
    return num1+num2;
}
int sub(int num1, int num2){
    return num1-num2;
}

typedef int (*fptrOperation)(int,int);
/*
int compute(fptrOpertation operation, int num1, int num2){
    return operation(num1, num2);
}

fptrOpertation select(char opcode){
    switch(opcode){
        case '+':return add;
        case '-':return sub;
    }
}

int evaluate(char opcode, int num1, int num2){
    fptrOpertation operation = select(opcode);
    return operation(num1,num2);
}
*/

typedef int (*operation)(int,int);
operation operations[128] = {NULL};

void init(){
    operations['+'] = add;
    operations['-'] = sub;
}

int evaluateArray(char opcode, int num1, int num2){
    fptrOperation operation;
    operation = operations[opcode];
    return operation(num1,num2);
}

int main(){
    init();
    printf("%d",evaluateArray('+',4,5));
}