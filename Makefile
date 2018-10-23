CC = CC
CFLAGS = -dynamic -std=c++11 -O3 -mkl
INCLUDES = -I/usr/common/software/intelcaffe/0.9999/intel/include -I/global/homes/a/alexr/DL267 -I/usr/common/software/boost/1.61/hsw/intel/include -I/usr/common/software/glog/0.3.4/intel/include -I/usr/common/software/gflags/2.1.2/intel/include 
#-I/usr/common/software/mkl/11.3/include
#~/caffe/src/caffe/proto/caffe.proto
# lopencv_core -lopencv_highgui -lopencv_imgproc
LFLAGS = -L/usr/common/software/boost/1.61/hsw/intel/lib -L/usr/common/software/intelcaffe/0.9999/intel/lib -L/usr/common/software/glog/0.3.4/intel/lib
LIBS = -lboost_system -lglog -lcaffe
#  -lblas
SRCS = mpi_calling.cpp read_cifar.cpp read_file.cpp timer.cpp
OBJS = $(SRCS:.c=.o)

MAIN = mpi_calling

.PHONY: depend clean

all: $(SOURCES) $(MAIN)
	@echo MAIN has been compiled

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

.c.o: $(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

#clean:
#	$(RM) *.o *~ $(MAIN) 

clean:
	rm -f $(OBJS) $(MAIN)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

