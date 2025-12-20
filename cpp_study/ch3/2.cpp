#include <iostream>
using namespace std;
// 참조자 &
int main(){
    int a = 3;
    int& another_a = a; // 또 다른 이름

    another_a = 5;
    cout << "a: " << a << endl;
    cout << "another_a : " << another_a << endl;
    return 0;
}
