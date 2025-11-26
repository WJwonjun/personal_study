#define MAX 10
struct stack{
    int top;
    int data[MAX];
};

typedef struct stack stackType;

void init(stackType* sp);

int is_empty(stackType* sp);
int is_full(stackType* sp);

void push(stackType* sp, const int item);
int pop(stackType* sp);