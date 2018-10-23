#ifndef READ_CIFAR
#define READ_CIFAR

#include "read_file.h"
#include <fstream>
class read_cifar : public read_file{
    public: read_cifar(){
                n_rows = 32;
                n_cols = 32;
                train_size = 50000;
                test_size = 10000;
                n_channels = 3;
            }
    protected: void read_data(std::string dir, float* temp_images, float* temp_labels,
                       float* test_images, float* test_labels);
               void read_image(std::ifstream& file, char* labelBuffer, char* imageBuffer,
                       float* labelTarget, float* imageTarget);
};
#endif
