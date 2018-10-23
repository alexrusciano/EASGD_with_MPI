#include <read_cifar.h>
#include <sstream>
#include <stdexcept>
//expects data in cifar10_train, cifar10_test

void read_cifar::read_image(std::ifstream& ifs, char* labelBuffer, char* imageBuffer, 
        float* labelTarget, float* imageTarget){
    ifs.read(labelBuffer,1);
    ifs.read(imageBuffer, n_channels*n_rows*n_cols);
    labelTarget[0] = (float) (unsigned char) labelBuffer[0];
    for(int i = 0; i < n_channels*n_rows*n_cols; i++){
        imageTarget[i] = ((float) (unsigned char) imageBuffer[i])/256.0;
    }
}

template <typename T>
std::string ToString(T val){
    std::stringstream stream;
    stream << val;
    return stream.str();
}

void read_cifar::read_data(std::string full_path,float* temp_images, float* temp_labels,
        float* test_images, float* test_labels){
    char imageBuffer[n_channels*n_rows*n_cols];
    char labelBuffer[1]; 
    std::string file_name;
    for(int k = 0; k < 5; k++){
        file_name = full_path + "/cifar-10-batches-bin/data_batch_" 
            + ToString(1+k) + ".bin";
        std::ifstream file(file_name.c_str(), std::ios::binary);
       if (file.is_open()){
            for(int i=0; i < 10000; i++){
                read_image(file, labelBuffer, imageBuffer, 
                        temp_labels + i+k*10000, temp_images + (k*10000+i)*n_channels*n_rows*n_cols);
                }
        }else{
            throw std::runtime_error("Can't open image file " + file_name + "!");
        }
        file.close();
    }
        // let's do the same thing for the train set
    file_name = full_path + "/cifar-10-batches-bin/test_batch.bin";
    std::ifstream file(file_name.c_str(), std::ios::binary);
    if (file.is_open()){
        for(int i=0; i < test_size; i++){
            read_image(file, labelBuffer, imageBuffer, 
                    test_labels + i, test_images + n_channels*n_rows*n_cols*i);
            }
    }else{
        throw std::runtime_error("Can't open image file " + file_name + "!");
    }
    file.close();
}
