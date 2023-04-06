// Ellen test
//compile when makefile is being weird: 
///usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L /usr/local/cuda/lib64 -lcuda -lcudart -lm -o EllenTest ellentest.cu


#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
//#include <complex.h> //for complex numbers

//testing
__global__ void hermitian_transpose_kernel(const float2* input_h, float2* output_hh, int N) { //const because we do not want to modify the input matrix!!!
	int col = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int row = threadIdx.y + blockDim.y * blockIdx.y;

    if (col < N && row < N) {
        int idx_in = col + row * N; //what index we are on in matrix
        int idx_out = row + col * N; //output should be reversed (transpose)

	//conjugate here - in a float2: .x is the real part, .y is imaginary part
        output_hh[idx_out].x = input_h[idx_in].x; //conjugate
        output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
    }
}

//A size (K, M)
//B size (K, N)
//C size ((N or K), M)
//B*A = C dont know why is flipped
__global__ void complex_matrix_mult_kernel(const float2* A, const float2* B, float2* C, const int M, const int K, const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float2 sum = make_float2(0.0f, 0.0f);

        for (int k = 0; k < K; k++) {
            float2 a = A[row * K + k];
            float2 b = B[k * N + col];

            float real_part = a.x * b.x - a.y * b.y;
            float imag_part = a.x * b.y + a.y * b.x;

            sum.x += real_part;
            sum.y += imag_part;
        }

        C[row * N + col] = sum;
    }
}

int main(int argc, char *argv[])  {

/*
* ellen test
*/

	int Grid_Dim_x=1, Grid_Dim_y=1;			//Grid structure values
	int Block_Dim_x=1, Block_Dim_y=1;		//Block structure values

	int noThreads_x, noThreads_y;		// number of threads available in device, each dimension
	int noThreads_block;				// number of threads in a block

	int N = 2;  					// size of array in each dimension

	float2 h[N*N] = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f} };
	float2 y[N] =  { {1.0f, 3.0f}, {4.0f, 8.0f} };//2x1 vector
	
	for (int i = 0; i < N*N; i++) { //print input matrix
		printf("(%f + %fi)\n", h[i].x, h[i].y);
	}
	
	float2 *mat_h,*mat_hh,*mat_hhh, *vec_y, *vec_hy; //float2 z = {1.0f, 2.0f}; // z = 1.0 + 2.0i
	int size;					// number of bytes in arrays

	cudaEvent_t start, stop;     		// using cuda events to measure time
	float elapsed_time_ms;       		// which is applicable for asynchronous code also

/* --------------------ENTER INPUT PARAMETERS AND DATA -----------------------*/
		
	Grid_Dim_x = 1;
	Grid_Dim_y = 1;
	Block_Dim_x = 2;
	Block_Dim_y = 2;

	noThreads_x = Grid_Dim_x * Block_Dim_x;		// number of threads in x dimension
	noThreads_y = Grid_Dim_y * Block_Dim_y;		// number of threads in y dimension

	noThreads_block = Block_Dim_x * Block_Dim_y;	// number of threads in a block

	dim3 Grid(Grid_Dim_x, Grid_Dim_x);		//Grid structure
	dim3 Block(Block_Dim_x,Block_Dim_y);	//Block structure, threads/block limited by specific device

	size = N * N * sizeof(float2);		// number of bytes in total in arrays

/* ------------- COMPUTATION DONE ON GPU ----------------------------*/

	cudaMalloc((void**)&mat_h, size);		// allocate memory on device
	cudaMalloc((void**)&mat_hh, size);
	cudaMalloc((void**)&mat_hhh, size);
	cudaMalloc((void**)&vec_y, (N*sizeof(float2))); //size of y vector is Nx1
	cudaMalloc((void**)&vec_hy, (N*sizeof(float2))); //size of Hy vector is Nx1

	cudaMemcpy(mat_h, h, size ,cudaMemcpyHostToDevice); //put h in device
	cudaMemcpy(vec_y, y, (N*sizeof(float2)) ,cudaMemcpyHostToDevice); //put y in device

//--------------------------TRANSPOSE-Hh---------------------------------
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	hermitian_transpose_kernel<<<Grid,Block>>>(mat_h,mat_hh,N); //calc hermitian Hh

	float2 output[N*N];//just to print, device has mat_hh, host does not need it?
	cudaMemcpy(output, mat_hh, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     	// instrument code to measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

	for (int i = 0; i < N*N; i++) {
		printf("(%f + %fi)\n", output[i].x, output[i].y);
	}
	
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // print out execution time

//-------------------------MATMUL-HhH--------------------------------------
	//a is H, output is Hh
	//this is HhH
	complex_matrix_mult_kernel<<<Grid,Block>>>(mat_h, mat_hh, mat_hhh, N,N,N);//(A, const float2* B, float2* C, const int M, const int K, const int N)

	float2 gramian[N*N];
	cudaMemcpy(gramian, mat_hhh, size, cudaMemcpyDeviceToHost);

	printf("gramian\n");
	for (int i = 0; i < N*N; i++) {
		printf("(%f + %fi)\n", gramian[i].x, gramian[i].y);
	}
	
//-------------------------MAT-VEC-MUL-Hy--------------------------------------
	//this is Hhy
	complex_matrix_mult_kernel<<<Grid,Block>>>(vec_y, mat_hh, vec_hy, 1,N,N);
	
	float2 hy[N];
	cudaMemcpy(hy, vec_hy, (N*sizeof(float2)), cudaMemcpyDeviceToHost);
	
	printf("Hy\n");
	for (int i = 0; i < N; i++) {
		printf("(%f + %fi)\n", hy[i].x, hy[i].y);
	}

/* --------------  clean up  ---------------------------------------*/
	cudaFree(mat_h);
	cudaFree(mat_hh);
	cudaFree(mat_hhh);
	cudaFree(vec_y);
	cudaFree(vec_hy);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
