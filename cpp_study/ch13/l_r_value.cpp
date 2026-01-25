int& func1(int& a) { return a; }
int func2(int b) { return b; }
//좌측값: &(주소) 구할 수 있는 타입
//우측값: 주소 없는 경우, 나타났다 사라짐
int main() {
int a = 3;
func1(a) = 4; // func1은 좌측값 레퍼런스 리턴 -> 선언 가능
std::cout << &func1(a) << std::endl;
int b = 2;
a = func2(b); // 가능
func2(b) = 5; // 오류 1: func2는 우측값(int) 리턴 -> 왼쪽 값 불가능
std::cout << &func2(b) << std::endl; // 오류 2: 우측값의 주소는 취할 수 없음
}