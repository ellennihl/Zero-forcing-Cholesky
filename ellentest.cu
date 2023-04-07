// Ellen test
//compile when makefile is being weird: 
///usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L /usr/local/cuda/lib64 -lcuda -lcudart -lm -o EllenTest ellentest.cu

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
//#include <complex.h> //for complex numbers

//testing xy 4*2
__global__ void hermitian_transpose(const float2* input_h, float2* output_hh, int N, int K) { //const because we do not want to modify the input matrix!!!
	int col = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int row = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3

    if (col < N && row < K) {
		//translate from ex 1,3 to index 1+3*2 = 7
        int idx_in = col + row * N; //what index we are on in matrix
		//1,3 to instead 3,1 : index 3+1*4=7
        int idx_out = row + col * K; //output should be reversed (transpose)

	//conjugate here - in a float2: .x is the real part, .y is imaginary part
        output_hh[idx_out].x = input_h[idx_in].x; //conjugate
        output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
    }
}

//A size (K, M)
//B size (K, N)
//C size ((N or K), M)
//B*A = C dont know why is flipped
/*__global__ void complex_matrix_mult(const float2* A, const float2* B, float2* C, const int M, const int K, const int N) {
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
}*/
//axb * cxd = axd
//b=c otherwise matmul cant happen
//K*M * N*1 = K*1
__global__ void complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_row_b_col, const int res_col) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < res_row && col < res_col) {
        float2 sum = make_float2(0.0f, 0.0f);

        for (int k = 0; k < a_row_b_col; k++) {
            float2 a = A[row * a_row_b_col + k]; //
            float2 b = B[k * res_col + col];

            float real_part = a.x * b.x - a.y * b.y;
            float imag_part = a.x * b.y + a.y * b.x;

            sum.x += real_part;
            sum.y += imag_part;
        }

        C[row * res_col + col] = sum;
		//if column done (col == K)- set event for cholesky?
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

	int N = 4;  		//antennas
	int K = 2;			//users

	float2 h[N*K] = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}, {13.0f, 14.0f}, {15.0f, 16.0f} };
	float2 y[N] =  { {1.0f, 3.0f}, {4.0f, 8.0f}, {16.0f, 8.0f}, {2.0f, 2.0f} };//2x1 vector
	//is y complex or not?
	
	for (int i = 0; i < N*K; i++) { //print input matrix
		printf("(%f + %fi)\n", h[i].x, h[i].y);
	}
	
	float2 *mat_h,*mat_hh,*mat_hhh, *vec_y, *vec_hy; //float2 z = {1.0f, 2.0f}; // z = 1.0 + 2.0i
	int mat1_size;					// number of bytes in arrays
	int mat2_size;
	int vec1_size;
	int vec2_size;
	
	cudaEvent_t start, stop;     		// using cuda events to measure time
	float elapsed_time_ms;       		// which is applicable for asynchronous code also

/* --------------------ENTER INPUT PARAMETERS AND DATA -----------------------*/
		
	Grid_Dim_x = 1;
	Grid_Dim_y = 1;
	Block_Dim_x = 4;
	Block_Dim_y = 2;

	noThreads_x = Grid_Dim_x * Block_Dim_x;		// number of threads in x dimension
	noThreads_y = Grid_Dim_y * Block_Dim_y;		// number of threads in y dimension

	noThreads_block = Block_Dim_x * Block_Dim_y;	// number of threads in a block

	dim3 Grid(Grid_Dim_x, Grid_Dim_x);		//Grid structure
	dim3 Block(Block_Dim_x,Block_Dim_y);	//Block structure, threads/block limited by specific device

	mat1_size = K * N * sizeof(float2); //Hh and H are K*N and N*K
	mat2_size = K * K * sizeof(float2);	//gramian is K*K
	vec1_size = N * sizeof(float2);		//vec is K*1
	vec2_size = K * sizeof(float2);

/* ------------- COMPUTATION DONE ON GPU ----------------------------*/

	cudaMalloc((void**)&mat_h, mat1_size);		// allocate memory on device
	cudaMalloc((void**)&mat_hh, mat1_size);
	cudaMalloc((void**)&mat_hhh, mat2_size);
	cudaMalloc((void**)&vec_y, vec1_size); //size of y vector is Nx1
	cudaMalloc((void**)&vec_hy, vec2_size); //size of Hy vector is Nx1

	cudaMemcpy(mat_h, h, mat1_size ,cudaMemcpyHostToDevice); //put h in device
	cudaMemcpy(vec_y, y, vec1_size ,cudaMemcpyHostToDevice); //put y in device

//--------------------------TRANSPOSE-Hh---------------------------------
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	hermitian_transpose<<<Grid,Block>>>(mat_h,mat_hh,N,K); //calc hermitian Hh

	float2 output[N*K];//just to print, device has mat_hh, host does not need it?
	cudaMemcpy(output, mat_hh, mat1_size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     	// instrument code to measure end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

	for (int i = 0; i < N*K; i++) {
		printf("(%f + %fi)\n", output[i].x, output[i].y);
	}
	
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // print out execution time

//-------------------------MATMUL-HhH--------------------------------------
	//a is H, output is Hh
	//this is HhH
	//complex_matrix_mult<<<Grid,Block>>>(mat_hh, mat_h, mat_hhh, K,N,K); //why no work
	complex_matrix_mult<<<Grid,Block>>>(mat_h, mat_hh, mat_hhh, K,N,K);//A, B, C, res_row, a_row_b_col, res_col

	float2 gramian[K*K];
	cudaMemcpy(gramian, mat_hhh, mat2_size, cudaMemcpyDeviceToHost);

	printf("gramian\n");
	for (int i = 0; i < K*K; i++) {
		printf("(%f + %fi)\n", gramian[i].x, gramian[i].y);
	}
	
//-------------------------MAT-VEC-MUL-Hy--------------------------------------
	//this is Hhy
	//complex_matrix_mult<<<Grid,Block>>>(mat_hh, vec_y, vec_hy, K,N,1); //why does this not work???
	complex_matrix_mult<<<Grid,Block>>>(vec_y, mat_hh, vec_hy, 1,N,K);//WHY IS IT FLIPPED
	
	float2 hy[N];
	cudaMemcpy(hy, vec_hy, vec2_size, cudaMemcpyDeviceToHost);
	
	printf("Hy\n");
	for (int i = 0; i < K; i++) {
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
