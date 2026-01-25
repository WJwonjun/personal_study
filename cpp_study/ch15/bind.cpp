#include <functional>
#include <iostream>
struct S {
int data;
S(int data) : data(data) { std::cout << "일반 생성자 호출!" << std::endl; }
S(const S& s) {
std::cout << "복사 생성자 호출!" << std::endl;
data = s.data;

}
S(S&& s) {
std::cout << "이동 생성자 호출!" << std::endl;
data = s.data;
}
};
void do_something(S& s1, const S& s2) { s1.data = s2.data + 3; }
int main() {
S s1(1), s2(2);
std::cout << "Before : " << s1.data << std::endl;

auto do_something_with_s1 =std::bind(do_something, std::ref(s1), std::placeholders::_1);
// ref: 명시적으로 s1의 레퍼런스를 전달 
// ref 없으면 그냥 s1 복사해버림
do_something_with_s1(s2);
std::cout << "After :: " << s1.data << std::endl;
}