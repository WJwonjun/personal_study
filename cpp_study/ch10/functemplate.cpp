#include <iostream>
#include <string>

template <typename T>
T max(T& a, T& b){
    return a > b? a:b;
}

template <typename Cont>
void bubble_sort(Cont& cont) {
for (int i = 0; i < cont.size(); i++) {
for (int j = i + 1; j < cont.size(); j++) {
if (cont[i] > cont[j]) {
cont.swap(i, j);
}
}
}
}

int main() {
int a = 1, b = 2;
std::cout << "Max (" << a << "," << b << ") ? : " << max(a, b) << std::endl;
std::string s = "hello", t = "world";
std::cout << "Max (" << s << "," << t << ") ? : " << max(s, t) << std::endl;
} // 클래스와는 다르게, max<int> 필요없음

