#include <iostream>
template <typename T> // 보편적 레퍼런스: 템플릿 인자 T에 대해 우측값과 좌측값 레퍼런스를 모두 받는 형태
void wrapper(T&& u) {
    g(std::forward<T>(u)); // forward: 우측값 레퍼런스에 대해 복사/생성 없이 우측값으로 전달
}
class A {};

void g(A& a) { std::cout << "좌측값 레퍼런스 호출" << std::endl; }
void g(const A& a) { std::cout << "좌측값 상수 레퍼런스 호출" << std::endl; }
void g(A&& a) { std::cout << "우측값 레퍼런스 호출" << std::endl; }
int main() {
A a;
const A ca;
std::cout << "원본 --------" << std::endl;
g(a);
g(ca);
g(A());
std::cout << "Wrapper -----" << std::endl;
wrapper(a);
wrapper(ca);
wrapper(A());
}