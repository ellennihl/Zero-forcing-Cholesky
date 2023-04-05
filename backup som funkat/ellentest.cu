// Ellen test

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
//#include <complex.h> //for complex numbers

__global__ void gpu_matrixadd(int *a,int *b, int *c, int N) {

	int col = threadIdx.x + blockDim.x * blockIdx.x; 
	int row = threadIdx.y + blockDim.y * blockIdx.y;

	int index = row * N + col;

      	if(col < N && row < N)
          c[index] = a[index]+b[index];

}

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

__global__ void gpu_matrixmult(int *gpu_a, int *gpu_b, int *gpu_c, int N) {

	int k, sum = 0;
	int col = threadIdx.x + blockDim.x * blockIdx.x; 
	int row = threadIdx.y + blockDim.y * blockIdx.y;

       if(col < N && row < N) {
		for(k = 0; k < N; k++) 
          		sum += gpu_a[row * N + k] * gpu_b[k * N + col];
		gpu_c[row * N + col] = sum;
	}
}

void cpu_matrixadd(int *a,int *b, int *c, int N) {

	int index;
	for(int col=0;col < N; col++) 
		for(int row=0;row < N; row++) {
			index = row * N + col;
           		c[index] = a[index]+b[index];
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
	//float2 *a,*b,*c,*d;
	float2 a[N*N] = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f} };
	
	for (int i = 0; i < N*N; i++) {
		printf("(%f + %fi)\n", a[i].x, a[i].y);
	}
	//float complex *mat_h,*mat_hh,*mat_hhh;
	float2 *mat_h,*mat_hh,*mat_hhh;
	//float2 z = {1.0f, 2.0f}; // z = 1.0 + 2.0i
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

	//a = (float2*) malloc(size);		//this time use dynamically allocated memory for arrays on host
	//b = (float2*) malloc(size);
	//c = (float2*) malloc(size);		// results from GPU
	//d = (float2*) malloc(size);		// results from CPU
	

/*	for(i=0;i < N;i++)			// load arrays with some numbers
	for(j=0;j < N;j++) {
		a[i * N + j] = i;
		b[i * N + j] = i;
	}*/
	//float2 a[N] = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f} };

/* ------------- COMPUTATION DONE ON GPU ----------------------------*/

	cudaMalloc((void**)&mat_h, size);		// allocate memory on device
	cudaMalloc((void**)&mat_hh, size);
	cudaMalloc((void**)&mat_hhh, size);

	cudaMemcpy(mat_h, a , size ,cudaMemcpyHostToDevice);
	//cudaMemcpy(mat_hh, b , size ,cudaMemcpyHostToDevice);
	//cudaMemcpy(mat_hhh, c , size ,cudaMemcpyHostToDevice);

	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
//	cudaEventSynchronize(start);  	// Needed?

	//gpu_matrixmult<<<Grid,Block>>>(mat_h,mat_hh,mat_hhh,N);
	hermitian_transpose_kernel<<<Grid,Block>>>(mat_h,mat_hh,N);

	float2 output[N*N];
	cudaMemcpy(output, mat_hh, size, cudaMemcpyDeviceToHost);

	//cudaMemcpy(c,mat_hhh, size ,cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

	for (int i = 0; i < N*N; i++) {
		printf("(%f + %fi)\n", output[i].x, output[i].y);
	}
	
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);  // print out execution time

/* ------------- COMPUTATION DONE ON HOST CPU ----------------------------*/
/*
	cudaEventRecord(start, 0);		// use same timing
//	cudaEventSynchronize(start);  	// Needed?

	cpu_matrixadd(a,b,d,N);		// do calculation on host

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );

	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms);  // print out execution time

/* ------------------- check device creates correct results -----------------*/
/*
	for(i=0;i < N*N;i++) {
		if (c[i] != d[i]) printf("*********** ERROR in results, CPU and GPU create different answers ********\n");
		break;
	}

	printf("\nEnter c to repeat, return to terminate\n");
	scanf("%c",&key);
	scanf("%c",&key);
*/

/* --------------  clean up  ---------------------------------------*/
//	free(a);
//	free(b);
//	free(c);
//	free(d);
	cudaFree(mat_h);
	cudaFree(mat_hh);
	cudaFree(mat_hhh);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}
