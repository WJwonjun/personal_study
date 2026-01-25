#include <functional>
#include <iostream>
#include <string>

int some_func1(const std::string& a) {
std::cout << "Func1 호출! " << a << std::endl;
return 0;
}

struct S {
void operator()(char c) { std::cout << "Func2 호출! " << c << std::endl; }
};

class A {
int c;
public:
A(int c) : c(c) {}
int some_func() {
std::cout << "비상수 함수: " << ++c << std::endl;
return c;
}
int some_const_function() const {
std::cout << "상수 함수: " << c << std::endl;
return c;
}
static void st() {}
};


int main() {
    std::function<int(const std::string&)> f1 = some_func1;
    std::function<void(char)> f2 = S();
    std::function<void()> f3 = []() { std::cout << "Func3 호출! " << std::endl; };
    f1("hello");
    f2('c');
    f3();

    A a(5);
    std::function<int(A&)> f4 = &A::some_func;
    std::function<int(const A&)> f5 = &A::some_const_function;
    f4(a);
    f5(a);
}