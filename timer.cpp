#include "timer.h"
#include <iostream>

void timer::start_compute(){
    this_start = clock();
}

void timer::start_communicate(){
    this_start = clock();
}

void timer::end_compute(){
    clock_t this_end = clock();
    compute_time += ((float) (this_end-this_start))/CLOCKS_PER_SEC;
}

void timer::end_communicate(){
    clock_t this_end = clock();
    communicate_time += ((float) (this_end-this_start))/CLOCKS_PER_SEC;
}

void timer::summary(){
    std::cout << "compute: " << compute_time << std::endl;
    std::cout << "communicate: " << communicate_time << std::endl;
}
