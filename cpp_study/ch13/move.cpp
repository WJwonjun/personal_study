#include <iostream>
#include <utility>

class A{
    public:
    A() { std::cout << "일반 생성자 호출!" << std::endl;}
    A(const A& a){ std::cout << "복사 생성자 호출!" << std::endl;}
    A(A&& a){ std::cout << "이동 생성자 호출!" << std::endl;}
};

int main(){
    A a;
    std::cout << "-----------" << std::endl;
    A b(a); // 여기선 a가 좌측값

    std::cout << "-----------" << std::endl;
    A c(std::move(a)); // 여기선 a가 우측값 (단순한 타입 변환만 시킴, 뭔가 이동 X)
}


// 이동만 활용해서 swap 수행하기
// MyString& MyString::operator=(MyString&& s) {
//     std::cout << "이동!" << std::endl;
//     string_content = s.string_content;
//     memory_capacity = s.memory_capacity;
//     string_length = s.string_length;
//     s.string_content = nullptr;
//     s.memory_capacity = 0;
//     s.string_length = 0;
//     return *this;
// }

// template <typename T>
//     void my_swap(T &a, T &b) {
//     T tmp(std::move(a));
//     a = std::move(b);
//     b = std::move(tmp);
// }