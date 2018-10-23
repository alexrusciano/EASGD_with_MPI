#include "read_file.h"
#include <mpi.h>
#include <random>
#include <time.h>
#include <cstring>

void read_file::read_distribute_file(std::string full_path, int world_rank, int num_p)
{
    float* temp_images;
    float* temp_labels;
    // proc 0 reads and informs others of dimensions
    // reading implemented below
    if(world_rank == 0){
        n_images = test_size;
        temp_images = new float[train_size*n_cols*n_rows*n_channels];
        temp_labels = new float[train_size];
        labels = new float[test_size];
        images = new float[train_size*n_channels*n_rows*n_cols];
        this->read_data(full_path, temp_images, temp_labels,
                images, labels);
    }else{
        temp_images = NULL;
        temp_labels = NULL;
        n_images = train_size;
        images = new float[train_size*n_channels*n_rows*n_cols];
        labels = new float[train_size];
    }
    // bcast.. we want to take advantage of our extra memory.
    if(world_rank ==0){
        MPI_Bcast(temp_images, train_size * n_channels*n_rows*n_cols,
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Bcast(images, train_size*n_channels*n_rows*n_cols, 
                MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(world_rank ==0){
        MPI_Bcast(temp_labels,train_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Bcast(labels, train_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    // and delete temp_images.. proc 0 doesn't train.
    if(world_rank ==0){
        delete[] temp_images;
        delete[] temp_labels;
    }
}

void read_file::permute_data(int world_rank){
    std::random_device rd;
    std::mt19937 eng(rd());
    std::uniform_int_distribution<> distr(0, n_images-1);
    float label_swap;
    float image_swap[n_cols*n_rows*n_channels];
    int shuffles = n_images*100;
    for (int i =0; i < shuffles; i++){
        int a = distr(eng);
        int b = distr(eng);
        label_swap = labels[b];
        labels[b] = labels[a];
        labels[a] = label_swap;
        std::memcpy(image_swap, images+n_rows*n_cols*n_channels*b,
                n_rows*n_cols*n_channels*sizeof(float));
        std::memcpy(images+n_rows*n_cols*n_channels*b, images+n_rows*n_cols*n_channels*a, 
                n_rows*n_cols*n_channels*sizeof(float));
        std::memcpy(images+n_rows*n_cols*n_channels*a, image_swap,
                n_rows*n_cols*n_channels*sizeof(float));
    }
}
