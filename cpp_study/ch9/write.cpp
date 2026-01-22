#include <fstream>
#include <iostream>
#include <string>

int main(){
    std::ofstream out("test.txt",std::ios::app);
    std::string s;
    if(out.is_open()){
        out << "저걸 쓰자~";
    }
    return 0;
}

// ios::binary
// ios::app