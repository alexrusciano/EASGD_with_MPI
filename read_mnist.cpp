#include "read_mnist.h"
#include <mpi.h>
void read_mnist::read_distribute_file(std::string full_path, int world_rank, int num_p)
{
    float* temp_images;
    // proc 0 reads and informs others of dimensions
    // reading implemented below
    if(world_rank == 0)
    {
        read_mnist::read_images(full_path, temp_images,test_images,
                n_images, n_test_images, n_cols, n_rows, image_size);
    }
    int a[4] = {n_images, n_cols, n_rows, image_size};
    MPI_Bcast(a, 4, MPI_INT, 0, MPI_COMM_WORLD);
    n_images = a[0], n_cols = a[1], n_rows = a[2], image_size = a[3];
    // scatter... proc 0 will get the extra few images
    if(world_rank ==0){
        train_images = NULL;
        MPI_Bcast(temp_images, n_images*28*28, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        train_images = new float[n_images*28*28];
        MPI_Bcast(train_images, n_images*28*28, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // and delete temp_images
    if(world_rank ==0)
    {
        delete[] temp_images;
    }

    // repeat for train labels
    float * temp_labels;
    // proc 0 reads and informs others of dimensions
    // reading implemented below
    if(world_rank == 0){
        read_mnist::read_labels(full_path, temp_labels, test_labels);
    }
    // scatter... proc 0 will get the extra few labels
    if(world_rank ==0){
        train_labels = NULL;
        MPI_Bcast(temp_labels, n_images, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        train_labels = new float[n_images];
        MPI_Bcast(train_labels, n_images, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // and delete temp_labels
    if(world_rank ==0)
    {
        delete[] temp_labels;
        n_images = 0;
    }
    // also set n_images to the local version
}

//helper for reading mnist
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8 ) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_mnist::read_images(std::string full_path,float* & temp_images, float*& test_images,
        int &n_images, int &n_test_images, int &n_cols, int &n_rows, int &image_size)
{
    std::string to_open = full_path + "/train-images-idx3-ubyte";
    std::ifstream file(to_open.c_str(), std::ios::binary);
        if (file.is_open())
        {
            int magic_number = 0;
            n_images=0;
            n_rows = 0;
            n_cols = 0;
            file.read((char*)&magic_number,sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            file.read((char*)&n_images, sizeof(n_images));
            n_images = reverseInt(n_images);
            file.read((char*)&n_rows,sizeof(n_rows));
            n_rows = reverseInt(n_rows);
            file.read((char*)&n_cols,sizeof(n_cols));
            n_cols = reverseInt(n_cols);
            image_size = n_rows *n_cols;
            temp_images = new float[n_images*image_size];
            char* image_buffer = new char[image_size];
            for(int i=0; i < n_images; ++i)
            {
                file.read(image_buffer, image_size);
                for(int j=0; j< image_size; ++j){
                    temp_images[i*image_size + j] = ((float) (unsigned char) image_buffer[j]) / 256.0;
                }
            }
            delete[] image_buffer;
        } else
        {
            throw std::runtime_error("Can't open image file " + full_path + "!");
        }
        file.close();

        // let's do the same thing for the train set
    std::string to_open_test = full_path + "/t10k-images-idx3-ubyte";
    std::ifstream test_file(to_open_test.c_str(), std::ios::binary);
        if (test_file.is_open())
        {
            int magic_number = 0;
            n_test_images = 0;
            n_rows = 0;
            n_cols = 0;
            test_file.read((char*)&magic_number,sizeof(magic_number));
            magic_number = reverseInt(magic_number);
            test_file.read((char*)&n_test_images, sizeof(n_test_images));
            n_test_images= reverseInt(n_test_images);
            test_file.read((char*)&n_rows,sizeof(n_rows));
            n_rows = reverseInt(n_rows);
            test_file.read((char*)&n_cols,sizeof(n_cols));
            n_cols = reverseInt(n_cols);
            image_size = n_rows *n_cols;
            test_images = new float[n_test_images*image_size];
            char* image_buffer = new char[image_size];
            for(int i=0; i < n_test_images; ++i)
            {
                test_file.read(image_buffer, image_size);
                for(int j=0; j< image_size; ++j){
                    test_images[i*image_size + j] = ((float) (unsigned char) image_buffer[j]) / 256.0;
                }
            }
            delete[] image_buffer;
        } else
        {
            throw std::runtime_error("Can't open image file " + full_path + "!");
        }
        test_file.close();
}

void read_mnist::read_labels(std::string full_path, float* & temp_labels, 
        float* &test_labels) 
{
    std::string to_open = full_path + "/train-labels-idx1-ubyte";
    std::ifstream file(to_open.c_str(), std::ios::binary);
    if (file.is_open())
    {
        int magic_number = 0;
        int number_of_labels=0;
        file.read((char*)&magic_number, sizeof(int));
        magic_number = reverseInt(magic_number);
        file.read((char*) & number_of_labels, sizeof(int)); 
        number_of_labels = reverseInt(number_of_labels);
        temp_labels = new float[number_of_labels];
        for(int i = 0; i < number_of_labels; i++)
        {
            unsigned char label_buffer;
            file.read((char*)&label_buffer, 1);
            temp_labels[i] = (float) label_buffer;
        }
    } else{
        throw std::runtime_error("Can't open label file " + full_path + "!");
    }
    file.close();
    std::string to_open_test = full_path + "/t10k-labels-idx1-ubyte";
    std::ifstream test_file(to_open_test.c_str(), std::ios::binary);
    if (test_file.is_open())
    {
        int magic_number = 0;
        int number_of_labels=0;
        test_file.read((char*)&magic_number, sizeof(int));
        magic_number = reverseInt(magic_number);
        test_file.read((char*) & number_of_labels, sizeof(int)); 
        number_of_labels = reverseInt(number_of_labels);
        test_labels = new float[number_of_labels];
        for(int i = 0; i < number_of_labels; i++)
        {
            unsigned char label_buffer;
            test_file.read((char*)&label_buffer, 1);
            test_labels[i] = (float) label_buffer;
        }
    } else{
        throw std::runtime_error("Can't open label file " + full_path + "!");
    }
    test_file.close();
}

