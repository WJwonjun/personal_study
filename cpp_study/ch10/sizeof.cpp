#include <iostream>

int sum_all(){return 0;}

template <typename... Ints>
// int sum_all(int num, Ints... nums){
//     return num + sum_all(nums...);
// }
int sum_all(Ints... nums){
    return (...+ nums); // fold식
}

template <typename... Ints>
double average(Ints... nums){
    return static_cast<double>(sum_all(nums...)) / sizeof...(nums);
}

int main(){
    std::cout << average(1,4,2,3,10) << std::endl;
}