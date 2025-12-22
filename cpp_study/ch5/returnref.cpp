#include <iostream>

class A{
    int x;

    public:
        A(int c): x(c) {}

        int& access_x(){return x;}
        int get_x(){return x;}
        void show_x(){std::cout << x << std::endl;}
};

int main(){
    A a(5);
    a.show_x();

    int& c = a.access_x();  // c = a의 x 

    c = 4;
    a.show_x();

    int d = a.access_x(); // d = 4;
    d = 3;
    a.show_x();

    int& e = a.get_x(); // 실행 끝나면 없어지는 int를 리턴하므로 x가 아닌 임시 생성자 x'를 받게 됨.
    e = 2;
    a.show_x();

    int f = a.get_x();
    f = 1;
    a.show_x();
}