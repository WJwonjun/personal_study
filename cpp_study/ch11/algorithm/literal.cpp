#include <iostream>
#include <string>
int main() {
std::string str = R"foo(
)"; <-- 무시됨
)foo";
std::cout << str;
}