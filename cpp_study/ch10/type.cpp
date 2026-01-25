#include <iostream>
#include <typeinfo>

template <int N>
struct Int {
static const int num = N;
};

template <typename T, typename U>
struct add {
typedef Int<T::num + U::num> result;
};

int main() {
    typedef Int<1> one;
    typedef Int<2> two;
    typedef add<one, two>::result three; // Int<3> 이 생성됨
    std::cout << "Addtion result : " << three::num << std::endl;
}

// 타입에 값 부여 or 연산 가능 : 템플릿 메타 프로그래밍