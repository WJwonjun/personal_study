#include <iostream>
#include <memory>
#include <vector>

class A : public std::enable_shared_from_this<A>{ // 제어 블록이 여러 개 생성되는 것을 막음
int *data;
public:
A() {
data = new int[100];
std::cout << "자원을 획득함!" << std::endl;
}
~A() {
std::cout << "소멸자 호출!" << std::endl;
delete[] data;
}
};

int main(){
    std::vector<std::shared_ptr<A>> vec;
    //vec.push_back(std::shared_ptr<A>(new A()));
    vec.push_back(std::make_shared<A>());
    vec.push_back(std::shared_ptr<A>(vec[0]));
    vec.push_back(std::shared_ptr<A>(vec[1]));

    // 벡터의 첫번째 원소를 소멸 시킨다.
    std::cout << "첫 번째 소멸!" << std::endl;
    vec.erase(vec.begin());
    // 그 다음 원소를 소멸 시킨다.
    std::cout << "다음 원소 소멸!" << std::endl;
    vec.erase(vec.begin());
    // 마지막 원소 소멸
    std::cout << "마지막 원소 소멸!" << std::endl;
    vec.erase(vec.begin());
    std::cout << "프로그램 종료!" << std::endl;
}

// shared_ptr은 주소값을 통해 생성하지 않는다. 