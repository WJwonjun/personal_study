#include <iostream>
#include <string>

class Employee {
protected: // 자식 클래스에서 접근 가능하도록 protected로 변경
    std::string name;
    int age;
    std::string position;
    int rank;

public:
    Employee(std::string name, int age, std::string position, int rank)
        : name(name), age(age), position(position), rank(rank) {}

    Employee(const Employee& employee) {
        name = employee.name;
        age = employee.age;
        position = employee.position;
        rank = employee.rank;
    }

    Employee() : name(""), age(0), position(""), rank(0) {}

    // 가상 함수(virtual)를 배우기 전이므로 일단 그대로 둡니다.
    virtual void print_info() {
        std::cout << name << " (" << position << " , " << age << ") ==> " << calculate_pay() << "만원" << std::endl;
    }
    virtual int calculate_pay() { return 200 + rank * 50; }
};

class Manager : public Employee {
    int year_of_service;

public:
    Manager(std::string name, int age, std::string position, int rank, int year_of_service)
        : Employee(name, age, position, rank), year_of_service(year_of_service) {}

    Manager(const Manager& manager)
        : Employee(manager), year_of_service(manager.year_of_service) {}

    Manager() : Employee(), year_of_service(0) {}

    int calculate_pay() override{ return 200 + rank * 50 + 5 * year_of_service; }

    void print_info() override{
        std::cout << name << " (" << position << " , " << age << ", "
                  << year_of_service << "년차) ==> " << calculate_pay() << "만원" << std::endl;
    }
};

class EmployeeList {
    int alloc_employee;
    int current_employee;
    Employee** employee_list; // 변수명 통일

public:
    EmployeeList(int alloc_employee) : alloc_employee(alloc_employee), current_employee(0), current_manager(0) {
        employee_list = new Employee * [alloc_employee];
    }

    void add_employee(Employee* employee) {
        if (current_employee < alloc_employee) {
            employee_list[current_employee++] = employee;
        }
    }

    int current_employ(){ return current_employ;}

    void print_employee_info() {
        int total_pay = 0;
        for (int i = 0; i < current_employee; i++) {
            employee_list[i]->print_info();
            total_pay += employee_list[i]->calculate_pay();
        }
        std::cout << "총 비용 : " << total_pay << "만원" << std::endl;
    }

    ~EmployeeList() {
        for (int i = 0; i < current_employee; i++) delete employee_list[i];
        delete[] employee_list;
    }
};

int main() {
    EmployeeList emp_list(10);
    emp_list.add_employee(new Employee("노홍철", 34, "평사원", 1));
    emp_list.add_employee(new Employee("하하", 34, "평사원", 1));
    emp_list.add_employee(new Manager("유재석", 41, "부장", 7));
    emp_list.add_employee(new Employee("정준하", 43, "과장", 4));
    emp_list.add_employee(new Manager("박명수", 43, "차장", 5));
    emp_list.add_employee(new Employee("정형돈", 36, "대리", 2));
    emp_list.add_employee(new Employee("길", 36, "인턴", -2));

    emp_list.print_employee_info();
    return 0;
}