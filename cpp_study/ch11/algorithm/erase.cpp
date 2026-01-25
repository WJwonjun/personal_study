#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <vector>
template <typename Iter>
void print(Iter begin, Iter end) {
    while (begin != end) {
    std::cout << "[" << *begin << "] ";
    begin++;
    }

    std::cout << std::endl;
}
struct is_odd {
    int* num_delete;
    is_odd(int* num_delete): num_delete(num_delete){}

    bool operator()(const int& i) { 
        if(*num_delete>=2) return false;
        if(i%2==1){
            (*num_delete)++;
            return true;
        }
        return false;
    }
};

int main() {
    std::vector<int> vec;
    vec.push_back(5);
    vec.push_back(3);
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);
    vec.push_back(4);

    std::cout << "처음 vec 상태 ------" << std::endl;
    print(vec.begin(), vec.end());

    std::cout << "홀수인 인 원소 제거 ---" << std::endl;
    int num_delete=0;
    //vec.erase(std::remove_if(vec.begin(), vec.end(), is_odd(&num_delete)), vec.end());
    vec.erase(std::remove_if(vec.begin(), vec.end(),
            [&num_delete](int i)  {
                if(num_delete>=2)
                    return false;
                else if(i%2==1){
                    num_delete++;
                    return true;
                }
                return false;
}),vec.end());
    // 람다 함수: [capture list] (받는인자) -> 리턴타입{ 함수본체}
    print(vec.begin(), vec.end());
}