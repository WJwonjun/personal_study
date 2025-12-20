#include <iostream>
#include <cmath>

class Point{
    int x, y;
public:
    Point(int pos_x, int pos_y) : x(pos_x), y(pos_y) {}
    Point(const Point& p) : x(p.x), y(p.y) {}

    int GetX() const { return x; }
    int GetY() const { return y; }
};

class Geometry{

    Point* point_array[100];
    int size;

public:
    Geometry() : size(0) {}

    Geometry(Point** point_list) : size(0) {
        int i = 0;
        while (point_list[i] != nullptr && size < 100) {
            point_array[size++] = new Point(*point_list[i]);
            i++;
        }
    }

    void AddPoint(const Point &point){
        if(size >= 100) return;
        point_array[size++] = new Point(point);
    }

    void PrintDistance() {
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                double dx = point_array[i]->GetX() - point_array[j]->GetX();
                double dy = point_array[i]->GetY() - point_array[j]->GetY();
                double dist = std::sqrt(dx * dx + dy * dy);

                std::cout << "Distance between point "
                          << i << " and " << j << " : "
                          << dist << std::endl;
            }
        }
    }

    void PrintNumMeets() {
        int count = 0;
        for (int i = 0; i < size; i++) {
            for (int j = i + 1; j < size; j++) {
                if (point_array[i]->GetX() == point_array[j]->GetX() &&
                    point_array[i]->GetY() == point_array[j]->GetY()) {
                    count++;
                }
            }
        }
        std::cout << "Number of meeting points: " << count << std::endl;
    }

    ~Geometry() {
        for (int i = 0; i < size; i++) {
            delete point_array[i];
        }
    }
};
