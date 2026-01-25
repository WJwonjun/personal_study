#include <algorithm>
#include <functional>
#include <iostream>
#include <vector>
using std::vector;

int main(){
    vector<int> a(1);
    vector<int> b(2);
    vector<int> c(3);
    vector<int> d(4);

    vector<vector<int>> container;
    container.push_back(b);
    container.push_back(d);
    container.push_back(a);
    container.push_back(c);

    vector<int> size_vec(4);
    
    //멤버 함수의 경우 transform에 곧바로 넣을 수 없으므로, std::function으로 변환 후 집어넣어준다.
    //std::function<size_t(const vector<int>&)> sz_func = &vector<int>::size;
    //std::transform(container.begin(), container.end(), size_vec.begin(),sz_func);
    
    //std::transform(container.begin(), container.end(), size_vec.begin(),std::mem_fn(&vector<int>::size));
    std::transform(container.begin(), container.end(), size_vec.begin(),[](const auto&v){ return v.size();});
    

    for (auto itr = size_vec.begin(); itr != size_vec.end(); ++itr) {
    std::cout << "벡터 크기 :: " << *itr << std::endl;
    }
}