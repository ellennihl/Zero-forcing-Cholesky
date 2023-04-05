// Matrix addition program MatrixMult.cu  Matrix multiplication
// written by Barry Wilkinson, UNC-Charlotte. Feb 2, 2011.

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>

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

void cpu_matrixmult(int *cpu_a, int *cpu_b, int *cpu_c, int N) {

	int row, col, k, sum;

	for(row=0; row < N; row++)   					// row of a
	  for(col=0; col < N; col++) {					// column of b
		sum = 0;
		for(k = 0; k < N; k++) 
          		sum += cpu_a[row * N + k] * cpu_b[k * N + col];
		cpu_c[row * N + col] = sum;
	  }
}

void printArray(int *h, int N) {

	printf("Array, every N/8 numbers, eight numbers, N => 8\n");

	for (int row = 0; row < N; row += N/8) {
	  for (int col = 0; col < N; col += N/8) 
	 	printf("%6d  ", h[col + row * N]);
	  printf("\n"); 
	}
}

void loadArrays(int *a, int *b, int N) {

	int row, col;

	srand(1);					// for repeatability
	for(row=0; row < N; row++)			// load arrays with some numbers
	   for(col=0; col < N; col++) {
		a[row * N + col] = rand() % 10;
		b[row * N + col] = rand() % 10;
	}
}

int main(int argc, char *argv[])  {

	char key;
	int i; 					// loop counter

	int Grid_Dim_x=1, Grid_Dim_y=1;	//Grid structure values
	int Block_Dim_x=1, Block_Dim_y=1;	//Block structure values

	int noThreads_x, noThreads_y;		// number of threads available in device, each dimension
	int noThreads_block;				// number of threads in a block

	int N = 10;  				// size of array in each dimension
	int *a,*b,*c,*d;
	int *dev_a, *dev_b, *dev_c;
	int size;					// number of bytes in arrays

	cudaEvent_t start, stop;     		// using cuda events to measure time
	float elapsed_time_ms1, elapsed_time_ms3;

/* --------------------ENTER INPUT PARAMETERS AND DATA -----------------------*/

do {  // loop to repeat complete program	

	printf ("Device characteristics -- some limitations (compute capability 2.x)\n");
	printf ("		Maximum number of threads per block = 1024\n");
	printf ("		Maximum sizes of x- and y- dimension of thread block = 1024\n");
	printf ("		Maximum size of each dimension of grid of thread blocks = 65535\n");
	
	printf("Enter size of array in one dimension (square array), currently %d\n",N);
	scanf("%d",&N);
	
	do {
		printf("Enter nuumber of blocks per grid in x and y dimensions, currently %d  : ",Grid_Dim_x);
		scanf("%d",&Grid_Dim_x);

		Grid_Dim_y = Grid_Dim_x;  // square grid

		printf("Enter nuumber of threads per block in x and y dimensions, currently %d (max 32): ",Block_Dim_x);
		scanf("%d",&Block_Dim_x);

		Block_Dim_y = Block_Dim_x;	//square blocks

		noThreads_x = Grid_Dim_x * Block_Dim_x;		// total number of threads in x dimension
		noThreads_y = Grid_Dim_y * Block_Dim_y;		// total number of threads in y dimension
		
		noThreads_block = Block_Dim_x * Block_Dim_y;	// number of threads in a block

		if (noThreads_x < N) printf("Error -- number of threads in x/y dimensions less than number of elements in arrays, try again\n");
		else if (noThreads_block > 1024) printf("Error -- too many threads in block, try again\n");
		else printf("Number of threads not used = %d\n", noThreads_x * noThreads_y - N * N);

	} while (noThreads_x < N || noThreads_block > 1024);

	dim3 Grid(Grid_Dim_x, Grid_Dim_x);		//Grid structure
	dim3 Block(Block_Dim_x,Block_Dim_y);	//Block structure, threads/block limited by specific device

/* ---------------- ALLOCATE MEMORY AND LOAD -------------------- */

	size = N * N * sizeof(int);		// number of bytes in total in arrays

	a = (int*) malloc(size);		//this time use dynamically allocated memory for arrays on host
	b = (int*) malloc(size);
	c = (int*) malloc(size);		// results from GPU
	d = (int*) malloc(size);		// results from CPU

	cudaMalloc((void**)&dev_a, size);	// allocate memory on device
	cudaMalloc((void**)&dev_b, size);
	cudaMalloc((void**)&dev_c, size);

	loadArrays(a,b,N);			// load arrays with numbers
	
	printf("Array A\n"); printArray(a, N); 
	printf("Array B\n"); printArray(b, N);

/* ------------- COMPUTATION DONE ON GPU ----------------------------*/

	cudaMemcpy(dev_a, a , size ,cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b , size ,cudaMemcpyHostToDevice);
	
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
//	cudaEventSynchronize(start);  	// Needed?

	gpu_matrixmult<<<Grid,Block>>>(dev_a,dev_b,dev_c,N);

	cudaMemcpy(c,dev_c, size ,cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms1, start, stop );
	
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms1);  // print out execution time

/* ------------- COMPUTATION DONE ON HOST CPU ----------------------------*/

	cudaEventRecord(start, 0);		// use same timing, seems necessary to do CPU after GPU otherwise time=0?
//	cudaEventSynchronize(start);  	// Needed? Put outside timing loop

	cpu_matrixmult(a,b,d,N);		// do calculation on host

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms3, start, stop );

	printf("Time to calculate results on CPU: %f ms.\n", elapsed_time_ms3);  // print out execution time

/* ------------------- check device creates correct results -----------------*/

	printf("\nArray C, as computed on CPU\n"); printArray(d, N); 
	printf("Checking all results the same\n");

	for(i=0;i < N*N;i++) {
		if (c[i] != d[i] ) { 
			printf("*********** ERROR in results, CPU and GPU create different answers ********\n");
			break;
		}
	}
/*--------------------------SPEEDUP ---------------------------------*/

	printf("Speedup on GPU compared to CPU= %f\n", (float) elapsed_time_ms3 / (float) elapsed_time_ms1); 
	
	printf("\nEnter c to repeat, return to terminate\n");
	scanf("%c",&key);
	scanf("%c",&key);

} while (key == 'c'); // loop of complete program

/* --------------  clean up  ---------------------------------------*/
	free(a);
	free(b);
	free(c);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}