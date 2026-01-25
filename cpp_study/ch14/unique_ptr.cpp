#include <iostream>
#include <memory>

class A {
    int *data;

    public:
A() {
std::cout << "자원을 획득함!" << std::endl;
data = new int[100];
}

void some() { std::cout << "일반 포인터와 동일하게 사용가능!" << std::endl; }

~A() {
std::cout << "자원을 해제함!" << std::endl;
delete[] data;
}
};

void do_something() {
std::unique_ptr<A> pa(new A());
std::cout << "pa : ";
pa->some();

std::unique_ptr<A> pb = std::move(pa); // pb에 소유권 이전
std::cout << "pb : ";
pb->some();
}

int main() { do_something(); }