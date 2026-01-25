#include <iostream>
#include <list>

int main(){
    std::list<int> lst;

    lst.push_back(10);
    lst.push_back(20);
    lst.push_back(30);
    lst.push_back(40);

    for(std::list<int>::iterator itr = lst.begin();itr!=lst.end();++itr){
        std::cout << *itr << std::endl;
    }
    // list의 iterator: 양방향으로 움직이되, 1칸씩밖에 못움직임
    // vector와 다르게, erase나 insert에도 iterator가 무효화되지 않음.
}