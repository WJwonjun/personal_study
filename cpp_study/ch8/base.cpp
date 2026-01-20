#include <iostream>
#include <string>

class Base {
    std::string s;
    public:
        Base() : s("기반") { std::cout << "기반 클래스" << std::endl; }
        virtual void what() { std::cout << s << std::endl; }
        // virtual: 실제 객체 종류 확인해보고 실행할 것. (동적 바인딩)
};

class Derived : public Base {
        std::string s;
    public:
        Derived() : s("파생"), Base() { std::cout << "파생 클래스" << std::endl; }
        void what() override{ std::cout << s << std::endl; }
        // 가상 함수를 오버라이드하고 있음을 알려줌.
};

int main() {
    Base p;
    Derived c;
    std::cout << "=== 포인터 버전 ===" << std::endl;
    Base* p_c = &c; // 자식을 부모로 선언하므로 부모에 대한 정보만 가져옴 (업캐스팅) <-> 다운 캐스팅은 하지 말자.
    p_c->what();
    return 0;
}