#ifndef READ_MNIST
#define READ_MNIST

#include "read_file.h"

class read_mnist : public read_file{
    public:
        void read_file(std::string, int, int);
        void read_images(std::string, float* &, float* &, int & , int &, int &, int &, int &);
        void read_labels(std::string, float* &, float* &);
        void read_distribute_file(std::string, int, int);
};

#endif
