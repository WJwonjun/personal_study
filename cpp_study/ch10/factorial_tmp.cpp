#include <iostream>

template <int N>
struct Factorial{
    static const int result = N * Factorial<N-1>::result;
};

template <>
struct Factorial<1>{
    static const int result=1;
};

int main(){
    std::cout << "6 != 1*2*3*4*5*6 = " << Factorial<6>::result << std::endl; 
}

// 컴파일러 타임에서 연산 끝내버리므로, 계산이 훨씬 빠름
// 실행 속도를 극한으로 끌어올리기 위해, CPU가 할 일을 컴파일러에게 미리 떠넘기는 기법