#include <iostream>

class Animal{ // 추상 클래스: 반드시 상속되어야 하는 클래스
    public:
        Animal(){}
        virtual ~Animal() {}
        virtual void speak() = 0; // 순수 가상 함수: 반드시 오버라이딩 되어야만 하는 함수
};

class Dog: public Animal{
    public:
        Dog(): Animal(){}
        void speak() override { std::cout << "왈왈" << std::endl; }
};

class Cat: public Animal{
    public:
        Cat(): Animal(){}
        void speak() override {std::cout << "야옹야옹" << std::endl;}
};

int main(){
    Animal* dog = new Dog();
    Animal* cat = new Cat();

    dog->speak();
    cat->speak();
}