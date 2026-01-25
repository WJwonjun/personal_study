#include <iostream>
#include <typeinfo>

int sum(int a, int b){ return a+b;}

class SomeClass{
    int data;

    public:
        SomeClass(int d): data(d){}
        SomeClass(const SomeClass& s): data(s.data){}
};

int main(){
    auto c = sum(1,2);
    auto num = 1.0+2.0;

    SomeClass some(10);
    auto some2 = some;
    auto some3(10);

    std::cout << "c 의 타입은? :: " << typeid(c).name() << std::endl;
    std::cout << "num 의 타입은? :: " << typeid(num).name() << std::endl;
    std::cout << "some2 의 타입은? :: " << typeid(some2).name() << std::endl;
    std::cout << "some3 의 타입은? :: " << typeid(some3).name() << std::endl;
}