#include <iostream>
#include <string>

void* operator new(std::size_t count) {
std::cout << count << " bytes 할당" << std::endl;
return malloc(count);
}

int main() {
std::cout << "string -----" << std::endl;
std::string s = "sometimes string is very slow";
std::cout << "--------------------" << std::endl;
std::cout << s.substr(0, 20) << std::endl << std::endl;
std::cout << "string_view -----" << std::endl;
std::string_view sv = s;
std::cout << "--------------------" << std::endl;
std::cout << sv.substr(0, 20) << std::endl;
}