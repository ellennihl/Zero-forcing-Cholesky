# NVCC is path to nvcc. Here it is assumed that /usr/local/cuda is on one's PATH.
NVCC = /usr/local/cuda/bin/nvcc
CUDAPATH = /usr/local/cuda

NVCCFLAGS = -I$(CUDAPATH)/include
LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lm

#VectorAdd:
#	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o VectorAdd VectorAdd.cu

MatrixAdd:
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o MatrixAdd MatrixAdd.cu

MatrixMult:
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o MatrixMult MatrixMult.cu

#Coalescing:
#	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o Coalescing Coalescing.cu


