Implementation of the synchronous version from the EASGD paper, 

<https://arxiv.org/abs/1412.6651> .

C++ interface of Caffe with MPI used for parallelism.  Dataset loaders included are MNIST and CIFAR10.

Compile with makefile and execute mpi_calling to run.  In running, one would specify 4 processes by entering

$ mpiexec -np 4 ./mpi_calling
