#ifndef TIMER
#define TIMER

#include <ctime>

class timer{
    public:
        float compute_time;
        float communicate_time;

        timer(){
            compute_time = 0;
            communicate_time = 0;
        }
        void start_compute();
        void start_communicate();
        void end_compute();
        void end_communicate();

        void summary();
    protected:
        clock_t this_start;
};

#endif
