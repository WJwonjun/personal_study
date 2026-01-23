#ifndef UTILS_H
#define UTILS_H

#include <string>
using namespace std;
namespace MyExcel{

class Table;
class Cell;

class Vector{
    string* data;
    int capacity;
    int length;

    public:
        Vector(int n=1);
        
        void push_back(string s);
        string operator[](int i);

        void remove(int x);

        int size();

        ~Vector();
};


class Stack {
    struct Node {
        Node* prev;
        string s;
        Node(Node* prev, string s) : prev(prev), s(s) {}
        };

    Node* current;
    Node start;

    public:
        Stack();

    void push(string s);
    
    string pop();
    string peek();
    bool is_empty();
    ~Stack();
};


class NumStack{
    struct Node {
        Node* prev;
        double s;
        Node(Node* prev, double s) : prev(prev), s(s) {}
        };
    Node* current;
    Node start;

    public:
        NumStack();
        void push(double s);
        double pop();
        double peek();
        bool is_empty();

        ~NumStack();
};

class Table{
    protected:
        int max_row_size, max_col_size;
        Cell*** data_table; // Cell*를 저장하는 2차원 배열
    
    public:
        Table(int max_row_size, int max_col_size);
        ~Table();

        void reg_cell(Cell* c, int row, int col);

        int to_numeric(const string& s);
        int to_numeric(int row, int col);

        string stringify(const string& s);
        string stringify(int row, int col);

        virtual string print_table() = 0;
};

class Cell{
    protected:
        int x,y;
        Table* table;
        string data;
    public:
        virtual string stringify();
        virtual int to_numeric();

        Cell(string data, int x, int y, Table* table);
};

class TxtTable: public Table{
    string repeat_char(int n, char c);

    string col_num_to_str(int n);

    public:
        TxtTable(int row, int col);

        string print_table();
};

class HtmlTable: public Table{
    public:
        HtmlTable(int row, int col);
        string print_table();
};

class CSVTable: public Table{
    public:
        CSVTable(int row, int col);
        string print_table();
};

}

#endif

