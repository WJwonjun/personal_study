#include <iostream>
class Parent{
    public:
        Parent() { std::cout << "Parent 생성자 호출" << std::endl;}
        virtual ~Parent() { std::cout << "Parent 소멸자 호출" << std::endl;}
    };

class Child: public Parent{
    public:
        Child(): Parent(){ std::cout << "Child 생성자 호출" << std::endl;}
        ~Child() { std::cout << "Child 소멸자 호출" << std::endl;}
};

int main(){
    std::cout << "평범한 Child" << std::endl;
    {
        Child c;
    }
    std::cout << "Parent 포인터로 Child 가리키기" << std::endl;
    {
        Parent *p = new Child();
        delete p;
    }
}

// Child 소멸자가 호출되면서 Parent의 소멸자도 호출