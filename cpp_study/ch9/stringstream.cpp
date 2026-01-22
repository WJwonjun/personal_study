#include <iostream>
#include <sstream>
#include <string>


double to_number(std::string s){
    std::istringstream ss(s);
    double x;

    ss >> x;
    return x;
}

int main(){
    // std::istringstream ss("123");
    // // 문자열을 하나의 스트림이라고 생각하게 해 주는 가상화 장치
    // int x;
    // ss >> x;

    // std::cout << "입력 받은 데이터:" << x << std::endl;
    std::cout << "변환:: 1 + 2 = " << to_number("1") + to_number("2") << std::endl;
    return 0;
}