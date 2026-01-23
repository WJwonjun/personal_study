#include "utils.h"
#include <iostream>
#include <fstream>   // ofstream 사용을 위해 반드시 필요
#include <algorithm> // max 함수 사용
#include <string>

namespace MyExcel {

// --- Vector 구현 ---
Vector::Vector(int n) : data(new string[n]), capacity(n), length(0) {}
void Vector::push_back(string s) {
    if (capacity <= length) {
        string* temp = new string[capacity * 2];
        for (int i = 0; i < length; i++) temp[i] = data[i];
        delete[] data;
        data = temp;
        capacity *= 2;
    }
    data[length++] = s;
}
string Vector::operator[](int i) { return data[i]; }
void Vector::remove(int x) {
    for (int i = x + 1; i < length; i++) data[i - 1] = data[i];
    length--;
}
int Vector::size() { return length; }
Vector::~Vector() { if (data) delete[] data; }

// --- Stack 구현 ---
Stack::Stack() : start(NULL, ""), current(&start) {}
void Stack::push(string s) { current = new Node(current, s); }
string Stack::pop() {
    if (current == &start) return "";
    string s = current->s;
    Node* prev = current;
    current = current->prev;
    delete prev;
    return s;
}
string Stack::peek() { return current->s; }
bool Stack::is_empty() { return current == &start; }
Stack::~Stack() { while (!is_empty()) pop(); }

// --- NumStack 구현 (데이터 타입 오류 수정) ---
// Node(prev, double) 생성자를 사용하도록 수정함
NumStack::NumStack() : start(NULL, 0.0), current(&start) {} 

void NumStack::push(double s) {
    // 임시 처리가 아닌 double 값을 직접 전달하는 정석적인 방식으로 수정
    current = new Node(current, s); 
}
double NumStack::pop() {
    if (current == &start) return 0;
    double val = current->s;
    Node* prev = current;
    current = current->prev;
    delete prev;
    return val;
}
double NumStack::peek() { return current->s; }
bool NumStack::is_empty() { return current == &start; }
NumStack::~NumStack() { while (!is_empty()) pop(); }

// --- Cell 구현 ---
Cell::Cell(string data, int x, int y, Table* table) : data(data), x(x), y(y), table(table) {}
string Cell::stringify() { return data; }
int Cell::to_numeric() { return 0; }

// --- Table 구현 ---
Table::Table(int max_row_size, int max_col_size) : max_row_size(max_row_size), max_col_size(max_col_size) {
    data_table = new Cell**[max_row_size];
    for (int i = 0; i < max_row_size; i++) {
        data_table[i] = new Cell*[max_col_size];
        for (int j = 0; j < max_col_size; j++) {
            data_table[i][j] = NULL;
        }
    }
}
Table::~Table() {
    for (int i = 0; i < max_row_size; i++) {
        for (int j = 0; j < max_col_size; j++) {
            if (data_table[i][j]) delete data_table[i][j];
        }
        delete[] data_table[i];
    }
    delete[] data_table;
}
void Table::reg_cell(Cell* c, int row, int col) {
    if (row < max_row_size && col < max_col_size) {
        if (data_table[row][col]) delete data_table[row][col];
        data_table[row][col] = c;
    }
}
int Table::to_numeric(const string& s) {
    int col = s[0] - 'A';
    int row = atoi(s.c_str() + 1) - 1;
    return to_numeric(row, col);
}
int Table::to_numeric(int row, int col) {
    if (row >= 0 && row < max_row_size && col >= 0 && col < max_col_size && data_table[row][col])
        return data_table[row][col]->to_numeric();
    return 0;
}
string Table::stringify(const string& s) {
    int col = s[0] - 'A';
    int row = atoi(s.c_str() + 1) - 1;
    return stringify(row, col);
}
string Table::stringify(int row, int col) {
    if (row >= 0 && row < max_row_size && col >= 0 && col < max_col_size && data_table[row][col])
        return data_table[row][col]->stringify();
    return "";
}

// --- TxtTable 구현 ---
TxtTable::TxtTable(int row, int col) : Table(row, col) {}
string TxtTable::print_table() {
    string total_table;
    int* col_max_wide = new int[max_col_size];
    for (int i = 0; i < max_col_size; i++) {
        unsigned int max_wide = 2;
        for (int j = 0; j < max_row_size; j++) {
            if (data_table[j][i] && data_table[j][i]->stringify().length() > max_wide)
                max_wide = data_table[j][i]->stringify().length();
        }
        col_max_wide[i] = (int)max_wide;
    }

    total_table += "    ";
    int total_wide = 4;
    for (int i = 0; i < max_col_size; i++) {
        int max_len = std::max(2, col_max_wide[i]);
        string col_name = col_num_to_str(i);
        total_table += " | " + col_name + repeat_char(max_len - col_name.length(), ' ');
        total_wide += (max_len + 3);
    }
    total_table += "\n";

    for (int i = 0; i < max_row_size; i++) {
        total_table += repeat_char(total_wide, '-') + "\n" + std::to_string(i + 1);
        total_table += repeat_char(4 - std::to_string(i + 1).length(), ' ');
        for (int j = 0; j < max_col_size; j++) {
            int max_len = std::max(2, col_max_wide[j]);
            string s = (data_table[i][j] ? data_table[i][j]->stringify() : "");
            total_table += " | " + s + repeat_char(max_len - s.length(), ' ');
        }
        total_table += "\n";
    }
    delete[] col_max_wide;
    return total_table;
}
string TxtTable::repeat_char(int n, char c) { string s = ""; for (int i = 0; i < n; i++) s += c; return s; }
string TxtTable::col_num_to_str(int n) {
    string s = "";
    if (n < 26) s += (char)('A' + n);
    else { s += (char)('A' + n / 26 - 1); s += (char)('A' + n % 26); }
    return s;
}

// --- HtmlTable 구현 ---
HtmlTable::HtmlTable(int row, int col) : Table(row, col) {}
string HtmlTable::print_table() {
    string s = "<table border='1' cellpadding='10'>";
    for (int i = 0; i < max_row_size; i++) {
        s += "<tr>";
        for (int j = 0; j < max_col_size; j++) {
            s += "<td>" + (data_table[i][j] ? data_table[i][j]->stringify() : "") + "</td>";
        }
        s += "</tr>";
    }
    s += "</table>";
    return s;
}

// --- CSVTable 구현 ---
CSVTable::CSVTable(int row, int col) : Table(row, col) {}
string CSVTable::print_table() {
    string s = "";
    for (int i = 0; i < max_row_size; i++) {
        for (int j = 0; j < max_col_size; j++) {
            if (j >= 1) s += ",";
            string temp = (data_table[i][j] ? data_table[i][j]->stringify() : "");
            // 큰따옴표 이스케이프 처리
            size_t pos = 0;
            while ((pos = temp.find('"', pos)) != string::npos) {
                temp.insert(pos, 1, '"');
                pos += 2;
            }
            s += "\"" + temp + "\"";
        }
        s += "\n";
    }
    return s;
}

// 연산자 오버로딩 (friend 혹은 전역)
std::ostream& operator<<(std::ostream& o, Table& table) {
    o << table.print_table();
    return o;
}

} // namespace MyExcel 끝

// --- 메인 함수 (범위 및 타입 오류 수정) ---
int main() {
    MyExcel::TxtTable table(5, 5);
    std::ofstream out("test.txt");

    // MyExcel:: 을 붙여 네임스페이스 내의 Cell을 명시함
    table.reg_cell(new MyExcel::Cell("Hello~", 0, 0, &table), 0, 0);
    table.reg_cell(new MyExcel::Cell("C++", 0, 1, &table), 0, 1);
    table.reg_cell(new MyExcel::Cell("Programming", 1, 1, &table), 1, 1);

    std::cout << std::endl << table;
    out << table;
    
    return 0;
}