#include <iostream>
#include <string>

void* operator new(std::size_t count){
    std::cout << count << " bytes 할당" << std::endl;
    return malloc(count);
}

int main(){
    std::cout << "s1 생성 --- " << std::endl;
    std::string s1 = "this is a pretty long sentence!!!";
    std::cout << "s1 크기 : " << sizeof(s1) << std::endl;
    std::cout << "s2 생성 --- " << std::endl;
    std::string s2 = "short sentence";
    std::cout << "s2 크기 : " << sizeof(s2) << std::endl;
}