#include <iostream>


class mystring{
    char* data;
    int len;
    int memory_capacity;

    public:
        explicit mystring(int capacity){
            data = new char[capacity];
            len = 0;
            memory_capacity = capacity;
        }
        mystring(char c){
            len = 1;
            data = new char[len+1];
            data[0] = c;
            data[1] = '\0';
        }
        mystring(const char* c){
            len = 0;
            while(c[len]!='\0') len++;
            data = new char[len+1];
            for(int i=0;i<len;i++){
                data[i] = c[i];
            }
            data[len] = '\0';
        }

        mystring(const mystring& str){
            len = str.len;
            data = new char[len+1];
            for(int i=0;i!=str.len;i++){
                data[i] = str.data[i];
            }
        }

        ~mystring(){
            delete[] data;
        }

        int strlen() const{
            return len;
        }

        int strlen(const char* str){
            int temp = 0;
            while(*str!='\0'){
                str++;
                temp++;
            }
            return temp;
        }

        mystring& insert(int loc, const mystring& str) {
            if (loc < 0 || loc > len) 
                return *this;

            // 메모리 부족 → 재할당
            if (len + str.len > memory_capacity) {

                if(memory_capacity*2> len+str.len) memory_capacity *=2;
                else memory_capacity = len+str.len;
                char* prev_data = data;
                data = new char[memory_capacity + 1];

                // 앞부분
                for (int i = 0; i < loc; i++)
                    data[i] = prev_data[i];

                // 삽입 문자열
                for (int j = 0; j < str.len; j++)
                    data[loc + j] = str.data[j];

                // 뒷부분
                for (int i = loc; i < len; i++)
                    data[i + str.len] = prev_data[i];

                delete[] prev_data;
            }
            else {
                // 뒤에서부터 밀기 (겹침 방지)
                for (int i = len - 1; i >= loc; i--)
                    data[i + str.len] = data[i];

                // 삽입
                for (int i = 0; i < str.len; i++)
                    data[loc + i] = str.data[i];
            }

            len += str.len;
            data[len] = '\0';
            return *this;
}


        mystring& insert(int loc, const char* str){
            mystring temp(str);
            return insert(loc, temp);
        }

        mystring& insert(int loc, char c){
            mystring temp(c);
            return insert(loc, temp);
        }

        mystring& erase(int loc, int num){
            if(num<0|| loc<0 || loc > len) return *this;

            
            if(loc+num>len){
                for(int i=loc;i<len;i++)
                    data[i] = '\0';
            }
            else{
            for (int i = loc + num; i < len; i++) {
            data[i - num] = data[i];
            }
        }
            len -= num;
            return *this;
        }

        mystring& assign(const mystring& str){
            if(str.len > memory_capacity){
                delete[] data;
                data = new char[str.len+1];
                memory_capacity = str.len;
            }
            for(int i=0;i<len;i++){
                data[i] = str.data[i];
            }

            len = str.len;

            return *this;
        }

        mystring& assign(const char* str){
            len = strlen(str);
            if(len > memory_capacity){
                delete[] data;
                data = new char[len+1];
                memory_capacity = len;
            }
            for(int i=0;i<len;i++){
                data[i] = str[i];
            }
            return *this;
        }

        bool strin(const mystring& key) const {
        for (int i = 0; i <= len - key.len; i++) {
            int j = 0;
            while (j < key.len && data[i + j] == key.data[j])
                j++;

            if (j == key.len)
                return true;
        }
        return false;
    }

        void print(){
            for(int i=0;i<len;i++){
                std::cout << data[i];
            }
            std::cout << std::endl;
        }

        int capacity(){return memory_capacity;}
        
        void reserve(int size){
            if(size>memory_capacity){
                char* prev_data = data;

                data = new char[size];
                memory_capacity = size;

                for(int i=0;i<len;i++){
                    data[i] = prev_data[i];
                }
                delete[] prev_data;
            }
        }

        char at(int i) const{
            if(i>=len || i<0) return '\0';
            else return data[i];
        }

        int find(int find_from, mystring& str) const{
            int i, j;
            if(str.len==0) return -1;
            for(i = find_from;i<len-str.len;i++){
                for(j=0;j<str.len;j++){
                    if(data[i+j]!=str.data[j]) break;
                }
                if(j==str.len) return i;
            }
            return -1;
        }


        int find(int find_from, const char* str) const{
            mystring temp(str);
            return find(find_from,temp);
        }
        int find(int find_from, char c) const{
            mystring temp(c);
            return find(find_from,temp);
        }

        int compare(const mystring& str) const{
            for(int i=0;i<std::min(len,str.len);i++){
                if(data[i]>str.data[i]) return 1;
                else if(data[i]<str.data[i]) return -1;
            }
            if(len==str.len) return 0;
            else if(len>str.len) return 1;
            return -1;
        }

};


int main(){
    mystring s(3);
}