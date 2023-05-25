#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>
//#include <omp.h>

/**
	This is to use qsort
*/
int cmpfunc (const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

/**
	Takes a csv containing a matrix and returns an array column major
*/
float2 *read_matrix_from_csv(char filename[], int num_rows, int num_cols) {
	// Allocate memory for the matrix
	float2 *matrix = (float2 *) malloc(num_rows * num_cols * sizeof(float2));
	bool real = true;

	char tempchar[20];
	strcpy(tempchar, filename);
	for(int i=0;i<2;i++){
		if(i == 0){
			strcat(filename, "_real.csv");
			real = true;
		}
		else{
			strcat(tempchar, "_imag.csv");
			strcpy(filename, tempchar);
			real = false;
		}
		
	  FILE *file = fopen(filename, "r");
	  if (file == NULL) {
		fprintf(stderr, "Error: Could not open file '%s'\n", filename);
		exit(1);
	  }
	  // Read the data from the file into the matrix
	  int size = (num_rows*20+1)*sizeof(char);//max 20 chars and "," for one value + "\0" or "\n"
	  char line[size];
	  int row = 0, col = 0;
	  while (fgets(line, size, file) && row < num_rows) {
		if (line[strlen(line) - 1] == '\n') {
		  line[strlen(line) - 1] = '\0';  // Remove newline character
		}

		char *token = strtok(line, ",");
		while (token != NULL && col < num_cols) {
			if(real){
				matrix[row + col * num_rows].x = atof(token); // Change the ordering of the matrix
			}
			else{
				matrix[row + col * num_rows].y = atof(token); // Change the ordering of the matrix
			}
			col++;
			token = strtok(NULL, ",");
		}
		col = 0;
		row++;
	  }
	  fclose(file);
	}
  
  return matrix;
}

/**
	extra calculates how many elements of a matrix each thread needs to calculate of there are too few threads
	elements is the number of elements there is in a row/column in the matrix.
	nrOfThreads are the number of threads available for use
*/
__inline__ __device__ int extra(int elements, int nrOfThreads){
	int tmp = ceil((float)elements/(float)nrOfThreads);
	return tmp;
}

/**
	cuCsqrt takes in a complex number and returns the square root of this number
	z the input complex number
	returns a complex number that is the square root of z
*/
__inline__ __device__ cuFloatComplex cuCsqrt(cuFloatComplex z){
	float r = cuCabsf(z);
    float theta = atan2(z.y,z.x);
    cuFloatComplex sqrt_z = make_cuFloatComplex(sqrtf(r) * cosf(theta / 2.0f),
                                                sqrtf(r) * sinf(theta / 2.0f));
	return sqrt_z;
}

/**
	This is the second stage of the matrix inverse.
	It takes the unfinished rows and subtract them with
	the multiplication of the ith column element in the row and the i row. 
	A is the matrix that is choleskylised
	i is the column that is calculated
	N is the nr of rows/columns of the A matrix (NxN)
	The A matrix is overwriten in this function
*/
__global__ void cInv2(float2* A,float2* Ainv, int i, int N){
	//for the column it is N elements.
	int rowElements = N-(i+1);
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(rowElements, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(N, colthread);
	
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	
	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
			int tmpRow = row+rowthread*v;
			int tmpCol = col+colthread*w;
			if(tmpRow <= rowElements && tmpCol <=N){
				if(tmpRow+i+1 >= tmpCol){
				Ainv[tmpCol*N+tmpRow+i+1] = cuCsubf(Ainv[tmpCol*N+tmpRow+i+1],cuCmulf(Ainv[tmpCol*N+i],A[i*N+tmpRow+i+1]));
				}
			}
		}
	}
	
    extraRows = extra(N, rowthread);

	int next_i = i+1;
	int index = 0;

	__shared__ float2 sharedDiagonal;
    for(int v=0;v < extraRows; v++){
        int tmpRow = row+rowthread*v;

        int idx = next_i * N + tmpRow + next_i; //find index
        int diagonal = (next_i * N) + next_i; //get diagonal element index

        

        if(idx == diagonal && tmpRow < N-next_i && next_i < N){//part 1, if diagonal element
            sharedDiagonal = cuCsqrt(A[idx]);
			index = idx;
        }
	}
        __syncthreads(); //every thread needs to reach this place before continuing execution
		A[index] = sharedDiagonal;
 for(int v=0;v < extraRows; v++){
        int tmpRow = row+rowthread*v;

        int idx = next_i * N + tmpRow + next_i; //find index
        int diagonal = (next_i * N) + next_i; //get diagonal element index    
    
		if(idx != diagonal && tmpRow < N-next_i && next_i < N){//part 2
            A[idx] = cuCdivf(A[idx], sharedDiagonal);//A[idx]/A[diagonal]
        }
    }
}

/**

	The first stage of column wise matrix inversion.
	In this stage the ith row is devided by its diagonal element
	
   A is the matrix that is choleskylised
   Ainv is the resulting inverted matrix and needs to be an empty matrix
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
*/
__global__ void cInv1(float2* A,float2* Ainv, int i, int N){
	int elements = i+1; 							//elements calculated
	int rowthread = blockDim.x * gridDim.x;			//nr of threads in the row
	int extraRows = extra(elements, rowthread);		//how many elements this thread will run 
	
	int col = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	for(int v=0;v < extraRows; v++){
		int tmpCol = col+rowthread*v;
		if(tmpCol <= elements){
			if(tmpCol == i){
				Ainv[tmpCol*N+i].x = 1;
			}
			Ainv[tmpCol*N+i] = cuCdivf(Ainv[tmpCol*N+i],A[i*N+i]);
		}
	}
}

/**
   The third step of the block cholesky decomposition where U-c*c^H.
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   The A matrix is overwriten in this function
*/
__global__ void bChol3(float2* A, float2* Ainv, int i, int N){
	
	int elements = i+1; 							//elements calculated
	int rowthread = blockDim.x * gridDim.x;			//nr of threads in the row
	int extraRows = extra(elements, rowthread);		//how many elements this thread will run 
	
	int col = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	for(int v=0;v < extraRows; v++){
		int tmpCol = col+rowthread*v;
		if(tmpCol <= elements){
			if(tmpCol == i){
				Ainv[tmpCol*N+i].x = 1;
			}
			Ainv[tmpCol*N+i] = cuCdivf(Ainv[tmpCol*N+i],A[i*N+i]);
		}
	}
	
	//here is inversion pt 2
	elements = N-(i+1);

	extraRows = extra(elements, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(elements, colthread);
	
	int j = i+1;
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	col = threadIdx.y + blockDim.y * blockIdx.y;
	

	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
				int tmpRow = row+rowthread*v;
				int tmpCol = col+colthread*w;
			if(tmpRow >= tmpCol && tmpRow<=elements && tmpCol<=elements){
				float2 tmp = A[(N*i+i+1)+tmpCol];
				tmp.y = -tmp.y;
				A[(tmpCol+j)*N+j+tmpRow] = cuCsubf(A[(tmpCol+j)*N+j+tmpRow],cuCmulf(A[(N*i+i+1)+tmpRow],tmp));
			}
		}
	}
}

/**
   The secons step of the block cholesky decomposition where c=c/d.
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   elements is the number of elements needed to calculate
   The A matrix is overwriten in this function
*/
__global__ void bChol2(float2* A,const int i,const int N){

    int rowthread = blockDim.x * gridDim.x;
    int extraRows = extra(N, rowthread);

    int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for

    for(int v=0;v < extraRows; v++){
        int tmpRow = row+rowthread*v;

        int idx = i * N + tmpRow + i; //find index
        int diagonal = (i * N) + i; //get diagonal element index

        __shared__ float2 sharedDiagonal;

        if(idx == diagonal  && tmpRow < N-i){//part 1, if diagonal element
            sharedDiagonal = cuCsqrt(A[idx]);
            A[idx] = sharedDiagonal;
        }
        __syncthreads(); //every thread needs to reach this place before continuing execution
        if(idx != diagonal && tmpRow < N-i){//part 2
            A[idx] = cuCdivf(A[idx], sharedDiagonal);//A[idx]/A[diagonal]
        }
    }
}


/**
   The first and second step of the block cholesky decomposition where sqrt(d) and c=c/d.
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   The A matrix is overwriten in this function
*/
__global__ void bChol(float2* A,int i,int N){
	A[i*N+i] = cuCsqrt(A[i*N+i]);
}

/**
	This metod takes in a matrix and returns the hermitian transpose of the matrix	
	input_h is the input matrix with size KxN
	output_hh is the resulting matrix with size NxK
	K is the nr of columns in input_h
	N the nr of rows in input_h
*/

__global__ void hermitian_transpose(const float2* input_h, float2* output_hh, int K, int N) { //const because we do not want to modify the input matrix!!!
	
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(N, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(K, colthread);
	
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
			int tmpRow = row+rowthread*v;
			int tmpCol = col+colthread*w;
			if(tmpRow < N && tmpCol < K){
				
				int idx_in = tmpCol * N + tmpRow;
				int idx_out = tmpRow * K + tmpCol;
				//conjugate here - in a float2: .x is the real part, .y is imaginary part
				output_hh[idx_out].x = input_h[idx_in].x; //conjugate
				output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
			}
		}
	}
}

/**
	Pre Condition: Same size at Arow/Bcol 
	This function calculates the dot product of two complex matrices where A.B=C
	A is the first input matrix
	B is the second input matrix
	C is the resulting matrix
	res_row is the nr of rows in matrix A
	a_col_b_row is the nr of columns of A matrix and nr of rows in B matrix
	res_col is nr of columns in B matrix
*/
__global__ void complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(res_row, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(res_col, colthread);

	int row = threadIdx.x + blockDim.x * blockIdx.x; 
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	for(int v=0;v < extraRows; v++){
			for(int w=0; w<extraCols;w++){
				int tmpRow = row+rowthread*v;
				int tmpCol = col+colthread*w;
				
				if (tmpRow < res_row && tmpCol < res_col) {		
					float2 sum = make_float2(0.0f, 0.0f);
				
				for (int k = 0; k < a_col_b_row; k++) {
					float2 a = A[k * res_row + tmpRow]; //column-major
					float2 b = B[tmpCol * a_col_b_row + k];
					float real_part = a.x * b.x - a.y * b.y;
					float imag_part = a.x * b.y + a.y * b.x;
					sum.x += real_part;
					sum.y += imag_part;
				}
				C[tmpCol * res_row + tmpRow] = sum;
			}
		}
	}
}

/**
	Pre Condition: Same size at Arow/Bcol 
	This function calculates the dot product of two complex matrices where A.B=C but only the lower tirangle
	A is the first input matrix
	B is the second input matrix
	C is the resulting matrix
	res_row is the nr of rows in matrix A
	a_col_b_row is the nr of columns of A matrix and nr of rows in B matrix
	res_col is nr of columns in B matrix
*/
__global__ void Ltriangle_complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {

	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(res_row, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(res_col, colthread);

	int row = threadIdx.x + blockDim.x * blockIdx.x; 
	int col = threadIdx.y + blockDim.y * blockIdx.y;


	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
			int tmpRow = row+rowthread*v;
			int tmpCol = col+colthread*w;
			if (tmpRow < res_row && tmpCol < res_col && tmpRow >= tmpCol) {		
				float2 sum = make_float2(0.0f, 0.0f);
				for (int k = 0; k < a_col_b_row; k++) {
					float2 a = A[k * res_row + tmpRow];
					float2 b = B[tmpCol * a_col_b_row + k];
					float real_part = a.x * b.x - a.y * b.y;
					float imag_part = a.x * b.y + a.y * b.x;
					sum.x += real_part;
					sum.y += imag_part;
				}
				C[tmpCol * res_row + tmpRow] = sum;
			}
		}
	}
}


__global__ void transpose_and_mult(const float2* input_h, float2* output_hh, int K, int N, const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col){
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(res_row, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(res_col, colthread);

	int row = threadIdx.x + blockDim.x * blockIdx.x; 
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	for(int v=0;v < extraRows; v++){
			for(int w=0; w<extraCols;w++){
				int tmpRow = row+rowthread*v;
				int tmpCol = col+colthread*w;
				
				if (tmpRow < res_row && tmpCol < res_col) {		
					float2 sum = make_float2(0.0f, 0.0f);
				
				for (int k = 0; k < a_col_b_row; k++) {
					float2 a = A[k * res_row + tmpRow]; //column-major
					float2 b = B[tmpCol * a_col_b_row + k];
					float real_part = a.x * b.x - a.y * b.y;
					float imag_part = a.x * b.y + a.y * b.x;
					sum.x += real_part;
					sum.y += imag_part;
				}
				C[tmpCol * res_row + tmpRow] = sum;
			}
		}
	}
	
	extraRows = extra(N, rowthread);
	extraCols = extra(K, colthread);
	
	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
			int tmpRow = row+rowthread*v;
			int tmpCol = col+colthread*w;
			if(tmpRow < N && tmpCol < K){
				
				int idx_in = tmpCol * N + tmpRow;
				int idx_out = tmpRow * K + tmpCol;
				//conjugate here - in a float2: .x is the real part, .y is imaginary part
				output_hh[idx_out].x = input_h[idx_in].x; //conjugate
				output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
			}
		}
	}
	
}

int main() {
	//read the Y.csv
	//128x8
	int K,N,blockSize,gridSize;
	//int K=128,N=4096,blockSize=32,gridSize=8;
	int nrOfFrames;
	
	printf("Enter N K blockSize gridSize nrOfFrames\n");
    scanf("%d %d %d %d %d",&N,&K,&blockSize,&gridSize,&nrOfFrames);
	//scanf("%d",&nrOfFrames);
	//int amount = 16;
	//int frameSet[amount] = {1,2,3,4,5,7,10,20,30,40,50,60,70,80,90,100,};
//for(int setOfFrames = 0; setOfFrames<amount;setOfFrames++){
	//nrOfFrames = frameSet[setOfFrames];
	
	printf("Info: %dx%d, blockSize=%d, gridSize=%d, nrOfFrames=%d\n",N,K,blockSize,gridSize,nrOfFrames);
	float elapsed_time_ms[10];       		// which is applicable for asynchronous code also
for(int loop=0;loop<10;loop++){	
	// read csv files
	char file1[32] = "";
	sprintf(file1, "%dx%d/Y", N,K);
	float2 *hY;
	hY = read_matrix_from_csv(file1, N, 1);
	
	float2 *H;
	sprintf(file1, "%dx%d/H", N,K);
	H = read_matrix_from_csv(file1, N, K);
	
	cudaStream_t *streams = (cudaStream_t *) malloc(nrOfFrames * sizeof(cudaStream_t));
	for(int frame = 0; frame < nrOfFrames; frame++){
		cudaStreamCreate(&streams[frame]);
	}

	
	cudaEvent_t start, stop;     		// using cuda events to measure time
	//float elapsed_time_ms;       		// which is applicable for asynchronous code also
	
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);
		
	//The h stands for host
	float2 *hHHY[nrOfFrames];


	for(int frame = 0; frame<nrOfFrames; frame++){
		hHHY[frame] = (float2 *) malloc(K * sizeof(float2));
	}
	
	//cudaEventRecord(start, 0);//start time +memcpy overhead
	//The d stands for device
	
	cuFloatComplex *dH[nrOfFrames], *dHH[nrOfFrames], *dmHH[nrOfFrames], *dInv[nrOfFrames], *dInvH[nrOfFrames],*dInvM[nrOfFrames],*dY[nrOfFrames],*dHHY[nrOfFrames],*dx[nrOfFrames];
	for(int frame = 0; frame<nrOfFrames; frame++){

		cudaMalloc((void **)&dH[frame], N*K*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dHH[frame],  K*N*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dmHH[frame], K*K*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dInv[frame], K*K*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dInvH[frame], K*K*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dInvM[frame], K*K*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dY[frame], N*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dHHY[frame], K*sizeof(cuFloatComplex));
		cudaMalloc((void **)&dx[frame], K*sizeof(cuFloatComplex));
	}	
		
	for(int frame=0; frame<nrOfFrames;frame++){
		//Copy input data to array on GPU.
		cudaMemcpyAsync(dH[frame], H, K*N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice,streams[frame]);
		cudaMemcpyAsync(dY[frame], hY, N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice,streams[frame]);
		
	}
	
	cudaEventRecord(start, 0);//start time
	//Run the transpose on gpu
	//Number of threads are N*K with N-rows and K-columns
	dim3 blockDims(blockSize,blockSize);
	dim3 GridDims(gridSize,gridSize);
	
	#pragma omp parallel for
	for(int frame = 0; frame<nrOfFrames; frame++){
		hermitian_transpose<<<GridDims,blockDims,0,streams[frame]>>>(dH[frame], dHH[frame],K,N);
	}

	#pragma omp parallel for
	for(int frame = 0; frame<nrOfFrames; frame++){
	//Number of threads are K*K
		Ltriangle_complex_matrix_mult<<<GridDims,blockDims,0,streams[frame]>>>(dHH[frame], dH[frame], dmHH[frame],K,N,K);	
	}
	
	int Block_Dim_x = K;
	int Block_Dim_y = 1;
	dim3 Block2(Block_Dim_x,Block_Dim_y);
	
	for(int i = 0; i < K; i++){	
		#pragma omp parallel for
		for(int frame = 0; frame<nrOfFrames; frame++){
			if(i==0){
				bChol2<<<1,Block2,sizeof(float2)*(1), streams[frame]>>>(dmHH[frame],i,K); //diagonal is in shared mem
			}
			bChol3<<<1,Block2, 0, streams[frame]>>>(dmHH[frame],dInv[frame],i,K);
			cInv2<<<1,Block2, sizeof(float2)*(1), streams[frame]>>>(dmHH[frame],dInv[frame],i,K);
		}

	}

	//#pragma unroll 10
	#pragma omp parallel for
	for(int frame = 0; frame<nrOfFrames; frame++){
		//This part takes the inv of L multiplied with itsef to become A^-1
		transpose_and_mult<<<GridDims,blockDims,0,streams[frame]>>>(dInv[frame], dInvH[frame],K,K, dHH[frame], dY[frame], dHHY[frame],K,N,1);
	}
	//cudaDeviceSynchronize();
	//#pragma unroll 10
	#pragma omp parallel for
	for(int frame = 0; frame<nrOfFrames; frame++){
		complex_matrix_mult<<<GridDims,blockDims,0,streams[frame]>>>(dInvH[frame], dInv[frame], dInvM[frame],K,K,K);
	}

	#pragma omp parallel for
	for(int frame = 0; frame<nrOfFrames; frame++){
		//dHH = 8x8 dHHY = 8x1
		complex_matrix_mult<<<GridDims,blockDims,0,streams[frame]>>>(dInvM[frame], dHHY[frame], dx[frame],K,K,1);		
	}
	
	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	for(int frame = 0; frame<nrOfFrames; frame++){
		cudaMemcpyAsync(hHHY[frame], dx[frame], K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost,streams[frame]);
	}
	cudaDeviceSynchronize();
	//cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms[loop], start, stop);

	for(int i =0; i<nrOfFrames; i++){
		cudaStreamSynchronize(streams[i]);
		
	}

	//PRINT RESULT_________________________________________________________
	if(loop==0 /*&& setOfFrames==1*/){
		printf("result x: -------------------------------------------------------------------------\n");
	float2 temp[K];
	//for (int frame = 0; frame<nrOfFrames; frame++){
		//printf("%d \n", i);
		memcpy(temp,hHHY[0],K*sizeof(cuFloatComplex));
		//temp = *hHHY[frame]; 
		for(int i = 0; i<K; ++i) {
			printf("%f+%fi \n", temp[i].x,temp[i].y);//(*hHHY[i*N + frame]).x,(*hHHY[i*N+frame]).y);
		}
	//}
	}
	
	
	
	    // Clean up CUDA streams
    for (int i = 0; i < nrOfFrames; ++i) {
        cudaStreamDestroy(streams[i]);
    }
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	
	// Free up the arrays on the GPU.
	cudaFree(dH);
	cudaFree(dHH);
	cudaFree(dmHH);
	cudaFree(dInv);
	cudaFree(dInvH);
	cudaFree(dInvM);
	cudaFree(dY);
	cudaFree(dHHY);
	
	//Free from CPU
	free(hY);
	free(H);
}
for(int i=0;i<10;i++)
		printf("Time to calculate results on GPU: %f ms or %f each.\n", elapsed_time_ms[i],elapsed_time_ms[i]/nrOfFrames);
//}
	return 0;
}
