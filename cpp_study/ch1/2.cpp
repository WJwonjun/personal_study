#include <iostream>
int sum(){
    int i,sum=0;
    for(int i=0;i<10;i++){
        sum+=i;
    }
    return sum;
}

int wh(){
    int i=1,sum=0;

    while(i<=10){
        sum+=i;
        i++;
    }
    return sum;
}
int main(){
    for(int i=0;i<10;i++){
        std::cout << i << std::endl;
    }
    std::cout << sum() << std::endl;
    std::cout << wh() << std::endl;
    return 0;
}
