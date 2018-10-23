#ifndef READ_FILE
#define READ_FILE

#include <iostream>
#include <string>

class read_file{
    public: void read_distribute_file(std::string dir, int,int);
            void permute_data(int world_rank);
    protected: virtual void read_data(std::string dir, float* temp_images, float* temp_labels,
                       float* test_images, float* test_labels) = 0;
    public: float* images;
            float* labels;
            int train_size;
            int test_size;
            int n_rows;
            int n_cols;
            int n_channels;
            int n_images;
};

#endif
