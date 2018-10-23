#define CPU_ONLY
#include <caffe/caffe.hpp>
#include "caffe/layers/memory_data_layer.hpp"
#include <mpi.h>
#include <iostream>
#include <string>
#include "read_cifar.h"
#include "timer.h"
using namespace caffe;
using namespace std;


boost::shared_ptr<Net<float> > ReadModelFromProto(const string& model_file)
{
    //we are on the CPU..
    Caffe::set_mode(Caffe::CPU);
    boost::shared_ptr<Net<float> > net_;
    // guess we have to free this later
    net_.reset(new Net<float>(model_file, TRAIN));
    CHECK_EQ(net_->num_inputs(),1 ) << "should have one input";
    CHECK_EQ(net_->num_outputs(),1) << "should have one output";
    return net_;
}

boost::shared_ptr<Solver<float> > ReadSolverFromProto(const string& solver_file)
{
    SolverParameter solver_param;
    ReadProtoFromTextFileOrDie(solver_file, &solver_param);
    boost::shared_ptr<Solver<float > > mySolver;
    mySolver.reset(SolverRegistry<float>::CreateSolver(solver_param));
    return mySolver;
}

// from francisr on github

void find_net_size(boost::shared_ptr<Net<float> > net, int & param_size){
    std::vector<boost::shared_ptr<Blob<float> > > net_params = net->params();
    param_size = 0;
    for (int i = 0; i < net_params.size(); ++i)
    {
        const Blob<float>* net_param = net_params[i].get();
        param_size += net_param->count();
    }
}

// copy network parameters into contiguous array, eases mpi message passing code.
void copy_params_from_net(boost::shared_ptr<Net<float> > net, float* params)
{
    std::vector<boost::shared_ptr<Blob<float> > > net_params = net->params();
    int offset = 0;
    for (int i = 0; i < net_params.size(); ++i)
    {
        Blob<float>* net_param = net_params[i].get();
        memcpy(params+offset, net_param->mutable_cpu_data(), sizeof(float)*net_param->count());
        //net_param->set_cpu_data(params + offset);
        offset += net_param->count();
    }
}

// change where network holds the param data.. I don't think caffe will move 
// them around in our
void copy_params_to_net(boost::shared_ptr<Net<float> > net, float* params)
{
    std::vector<boost::shared_ptr<Blob<float> > > net_params = net->params();
    int offset = 0;
    for (int i = 0; i < net_params.size(); ++i)
    {
        Blob<float>* net_param = net_params[i].get();
        net_param->set_cpu_data(params + offset);
        offset += net_param->count();
    }
}

//for convenience, at start we move all the network params to a contiguous array
void move_net_params(boost::shared_ptr<Net<float> > net, float* params){
    copy_params_from_net(net, params);
    copy_params_to_net(net, params);
}

// overwrites everyone's param_buffer
void update_global(boost::shared_ptr<Net<float> > myNet, float* param_buffer, float* myParams, 
        float alpha, float beta, int world_rank, int param_size){
    if(world_rank != 0){
        for (int i = 0; i < param_size; i++){
            param_buffer[i] = myParams[i] * alpha;
        }
    }else{
        for (int i = 0; i < param_size; i++){
            param_buffer[i] = myParams[i] * (1-beta);
        }
    }
    MPI_Reduce(param_buffer, myParams, param_size, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
}

// sets locals' param_buffer to global's myParam
void global_to_local(float* param_buffer, float* myParams, int param_size, int world_rank){
    if(world_rank ==0){
        MPI_Bcast(myParams, param_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }else{
        MPI_Bcast(param_buffer, param_size, MPI_FLOAT, 0,MPI_COMM_WORLD);
    }
}

// sets locals' myParam to correct update (Step() only adds gradient, so now we must add in the
// penalty for deviating global's myParams (stored as locals' param_buffer).
// ok, this is an efficiency issue, we really need to hold off on the gradient update in Step()
void update_local(boost::shared_ptr<Net<float> > myNet, float* param_buffer, 
        float* myParams, int param_size, float this_alpha){
    for(int i = 0; i < param_size; i++){
        myParams[i] += (param_buffer[i]- myParams[i]) * this_alpha;
    }
}

void attach_data(boost::shared_ptr<Net<float> > myNet, boost::shared_ptr<Net<float> > myTestNet, 
        read_file* data_holder, int world_rank){
    MemoryDataLayer<float> *dataLayer_trainnet;
    MemoryDataLayer<float> *dataLayer_testnet;
    dataLayer_trainnet = (caffe::MemoryDataLayer<float> *) (myNet->layer_by_name("data").get());
    dataLayer_testnet = (caffe::MemoryDataLayer<float> *)(myTestNet->layer_by_name("data").get());
    int train_batch_size = dataLayer_trainnet->batch_size();
    int test_batch_size = dataLayer_testnet->batch_size();
    dataLayer_trainnet->Reset(data_holder->images,data_holder->labels,
            data_holder->n_images / train_batch_size * train_batch_size);
    dataLayer_testnet->Reset(data_holder->images,data_holder->labels,
            data_holder->n_images / train_batch_size * train_batch_size);
} 
// params specified in main
// num_iter, num_iter_till_sync, elastic, learning_rate, num_p
// num_iter (not needed), num_iter_till_sync
//
//params specified in sgd_solver
// learning_rate
//
// params specified at run time
// num_p

int main(){
    int world_size;
    int world_rank;
    int param_size;
    float* param_buffer;
    float* myParams;
    read_cifar* data_holder = new read_cifar();
    const char* solver_file;
    boost::shared_ptr<Solver<float> > mySolver;
    boost::shared_ptr<Net<float> > myNet;
    boost::shared_ptr<Net<float> > myTestNet;
    timer* my_timer = new timer();
    // instantiate mpi
    MPI_Init(0,0);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    // instantiate solvers... protect read access since caffe doesn't?
    if (world_rank == 0){
        solver_file = "cifar_solver_global.prototxt";
    }else{
        solver_file = "cifar_solver.prototxt";
    }
    mySolver = ReadSolverFromProto(solver_file);
/*
    int my_turn;
    if(world_rank ==0){
        my_turn = 1;
    }else{
        my_turn = 0;
        MPI_Recv(&my_turn, 1, MPI_INT, world_rank-1, 0, MPI_COMM_WORLD, 0);
    }
    mySolver = ReadSolverFromProto(solver_file);
    if(world_rank != world_size-1){
        MPI_Send(&my_turn, 1, MPI_INT, world_rank+1, 0, MPI_COMM_WORLD);
    }
*/
    const int num_p = world_size-1;
    const int num_iter = 1501;
    const int num_iter_till_sync = 10;
    const float beta = .9/num_iter_till_sync;
    const float learning_rate = .05;
    const float alpha = beta / num_p;
    const float elastic = (alpha / learning_rate);

    MPI_Barrier(MPI_COMM_WORLD);
    myNet = mySolver->net();
    myTestNet = mySolver-> test_nets() [0];

    // I'm not sure about how / when test nets and train nets synchronize 
    // move params to contiguous array.. so mpi can do one broadcast each time
    // global_to_local will then broadcast the SAME initialization to all solvers..
    // the original paper suggests this is important
    find_net_size(myNet, param_size);
    param_buffer = new float[param_size];
    myParams = new float[param_size];
    
    move_net_params(myNet, myParams);
    global_to_local(myParams, myParams, param_size, world_rank);
    // distribute the data.  proc 0 gets all test data, others divide train data
    // mnist_holder
    data_holder->read_distribute_file("/global/cscratch1/sd/alexr", world_rank, world_size);
    data_holder->permute_data(world_rank);

    attach_data(myNet, myTestNet, data_holder, world_rank);
    //
    // TRAINING CODE
    //
    // big iteration loop:
    // update_local needs params_in from global's myParams
    // global needs params_buffer from local's 
    //
    
    global_to_local(param_buffer,myParams, param_size, world_rank);
    for( int cur_iter =0; cur_iter < num_iter; cur_iter += num_iter_till_sync){
        //step to next global comm
        if (world_rank != 0){
            for(int i = 0; i < num_iter_till_sync; i++){
                mySolver->Step(1);
            }
        }else{
            my_timer->start_compute();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if(world_rank == 0){
            my_timer->end_compute();
            my_timer->start_communicate();
        }
        // global comm
        update_global(myNet, param_buffer, myParams, alpha, beta, world_rank, param_size);
        global_to_local(param_buffer, myParams, param_size, world_rank);
        MPI_Barrier(MPI_COMM_WORLD);
       // for val error
        if(world_rank ==0){
            my_timer->end_communicate();
            std::cout << "cur_iter " << cur_iter << std::endl;
            mySolver->Step(1);
        }else{
            update_local(myNet, param_buffer, myParams, param_size, alpha);
            }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    delete data_holder;
    if(world_rank == 0){
        my_timer->summary();
    }
    delete my_timer;
    MPI_Finalize();
    std::cout <<"finalized process "<< world_rank << std::endl;
}

   //not used... shared_ptr<Net<float> > myNet = ReadModelFromProto(model_file);
   //std::vector<shared_ptr<Blob<float> > > net_params = myNet->params();
