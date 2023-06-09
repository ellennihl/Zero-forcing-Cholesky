#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>

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
	  char line[1024];
	  int row = 0, col = 0;
	  while (fgets(line, 1024, file) && row < num_rows) {
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
	cuCsqrt takes in a complex number and returns the square root of this number
	z the input complex number
	returns a complex number that is the square root of z
*/
__device__ cuFloatComplex cuCsqrt(cuFloatComplex z){
	float r = cuCabsf(z);
    float theta = atan2(z.y,z.x);
    cuFloatComplex sqrt_z = make_cuFloatComplex(sqrtf(r) * cosf(theta / 2.0f),
                                                sqrtf(r) * sinf(theta / 2.0f));
	return sqrt_z;
}

/**
	This is the second stage of the matrix inverse.
	A is the matrix that is choleskylised
	i is the column that is calculated
	N is the nr of rows/columns of the A matrix (NxN)
	The A matrix is overwriten in this funktion
*/
__global__ void cInv2(float2* A,float2* Ainv, int i, int N){
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	if(row+i+1 >= col){
		//printf("P2 (%d,%d) %d\n",row+i+1,col,i);
		//printf(" %d-%d*%d",j*N+k,j*N+i,i*N+k);
		Ainv[col*N+row+i+1] = cuCsubf(Ainv[col*N+row+i+1],cuCmulf(Ainv[col*N+i],A[i*N+row+i+1]));
		//Ainv[col*N+row] = Ainv[col*N+row]-Ainv[col*N+i]*A[i*N+row];
	}	
}

/**
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   The A matrix is overwriten in this funktion
*/
__global__ void cInv1(float2* A,float2* Ainv, int i, int N){
	int col = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	//printf("P1 (%d,%d) %d/%d \n",i,col,col*N+i,i*N+i);
	if(col == i){
		Ainv[col*N+i].x = 1;
	}
	Ainv[col*N+i] = cuCdivf(Ainv[col*N+i],A[i*N+i]);
	//printf("P1: %f,",Ainv[j*N+i]);	
}

/**
   The third step of the block cholesky decomposition where U-c*c^H.
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   The A matrix is overwriten in this funktion
*/
__global__ void bChol3(float2* A, int i, int N){
	int j = i+1;
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	//int vector = threadIdx.y + blockDim.y * blockIdx.y;
	if(row >= col){
		//printf("vec1 %d vec2 %d (%d,%d) index %d \n",(N*i+i+1)+row,(N*i+i+1)+col,row,col,(col+j)*N+j+row);
		//printf("%d = %d-%d*%d \n",(col+j)*N+j+row,(col+j)*N+j+row,(N*i+i+1)+row,(N*i+i+1)+col);
		float2 tmp = A[(N*i+i+1)+col];
		tmp.y = -tmp.y;
		A[(col+j)*N+j+row] = cuCsubf(A[(col+j)*N+j+row],cuCmulf(A[(N*i+i+1)+row],tmp));
	}	
}

/**
   The first and secons step of the block cholesky decomposition where sqrt(d) and c=c/d.
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   The A matrix is overwriten in this funktion
*/
__global__ void bChol2(float2* A,int i,int N){
	int row = blockIdx.x + 1;
	//int diagonal = i*N+i;
	/*if(row == 1){
		//printf("sqtr %d \n",i*N+i); this is a 
		A[i*N+i] = cuCsqrt(A[i*N+i]);
	}*/
	//__syncthreads();
	//printf("\n %d %d",i*N+i, (i*N+i)+row);
    A[(i*N+i)+row] = cuCdivf(A[(i*N+i)+row], A[i*N+i]);
}

/**
   The first and secons step of the block cholesky decomposition where sqrt(d) and c=c/d.
   A is the matrix that is choleskylised
   i is the column that is calculated
   N is the nr of rows/columns of the A matrix (NxN)
   The A matrix is overwriten in this funktion
*/
__global__ void bChol(float2* A,int i,int N){
	A[i*N+i] = cuCsqrt(A[i*N+i]);
}

/**
	This metod takes in a matrix and returns the hermetian transpose of the matrix	
	input_h is the input matrix with size NxK
	output_hh is the resulting matrix with size KxN
	N is the nr of columns in input_h
	K the nr of rows in input_h
*/
__global__ void hermitian_transpose(const float2* input_h, float2* output_hh, int K, int N) { //const because we do not want to modify the input matrix!!!
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	if (col < N && row < K) {
		int idx_in = col * K + row;
		int idx_out = row * N + col;
		//printf("(%d,%d)  in: %d, out: %d\n",row,col,idx_in,idx_out);
		//conjugate here - in a float2: .x is the real part, .y is imaginary part
        output_hh[idx_out].x = input_h[idx_in].x; //conjugate
        output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
    }
}

/**
	chould not exist chages the complex numbers to invert
	input_h is the input matrix with size NxK
	output_hh is the resulting matrix with size KxN
	N is the nr of in input_h
	K the nr of rows in input_h
*/
__global__ void complex_change(float2* input_h,int N) {
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	input_h[col*N+row].y = -input_h[col*N+row].y;
}

/**
	Pree Condition: Same size at Arow/Bcol 
	This funktion calculates the dot produkt of two complex matrixes where A.B=C
	A is the first input matrix
	B is the second input matrix
	C is the resulting matrix
	res_row is the nr of rows in matrix A
	a_col_b_row is the nr of columns of A matrix and nr of rows in B matrix
	res_col is nr of columns in B matrix
*/
__global__ void complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {

	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	if (row < res_row && col < res_col) {		
        float2 sum = make_float2(0.0f, 0.0f);

        for (int k = 0; k < a_col_b_row; k++) {
			//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);

            float2 a = A[k * res_row + row]; //column-major!!!!!!
            float2 b = B[col * a_col_b_row + k];
            float real_part = a.x * b.x - a.y * b.y;
            float imag_part = a.x * b.y + a.y * b.x;
            sum.x += real_part;
            sum.y += imag_part;
        }
		//if(row >= col){
			C[col * res_row + row] = sum;
		//}
	}
}

/**
	Pree Condition: Same size at Arow/Bcol 
	This funktion calculates the dot produkt of two complex matrixes where A.B=C but only the lower tirangle
	A is the first input matrix
	B is the second input matrix
	C is the resulting matrix
	res_row is the nr of rows in matrix A
	a_col_b_row is the nr of columns of A matrix and nr of rows in B matrix
	res_col is nr of columns in B matrix
*/
__global__ void Ltriangle_complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {

	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	if (row < res_row && col < res_col) {		
        float2 sum = make_float2(0.0f, 0.0f);

        for (int k = 0; k < a_col_b_row; k++) {
			//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);

            float2 a = A[k * res_row + row]; //column-major!!!!!!
            float2 b = B[col * a_col_b_row + k];
            float real_part = a.x * b.x - a.y * b.y;
            float imag_part = a.x * b.y + a.y * b.x;
            sum.x += real_part;
            sum.y += imag_part;
        }
		if(row >= col){
			C[col * res_row + row] = sum;
		}
	}
}

int main() {

	//read the Y.csv
	int num_rows = 128;
	int num_cols = 1;
	float2 *hY;
	char file[] = "128x8/Y";
	hY = read_matrix_from_csv(file, num_rows, num_cols);

	//read H.csv
	num_rows = 128;
	num_cols = 8;
	float2 *H;
	strcpy(file, "128x8/H");
	H = read_matrix_from_csv(file, num_rows, num_cols);
	/*
	printf("Matrix = \n");
	for (int i = 0; i<num_rows; ++i) {
		for (int j = 0; j<num_cols; ++j) {
			printf("%f+%fi ", H[j*num_rows+i].x,H[j*num_rows+i].y);
		}
        printf(";\n");
    }*/
	
	//Size of N=antennas (nr of rows), K=Users (nr of columns)
	int K = 128;
	int N = 8;

	//The h stands for host
	cuFloatComplex hHH[N*K], hmHH[N*N],hInv[K*K],hHHY[N];
	
	cudaEvent_t start, stop,start2,stop2;     		// using cuda events to measure time
	float elapsed_time_ms,elapsed_time_ms2;       		// which is applicable for asynchronous code also
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);
	cudaEventCreate(&start2);
	cudaEventCreate(&stop2);
		
	//The d stands for device
    cuFloatComplex *dH, *dHH, *dmHH, *dInv, *dInvH,*dInvM,*dY,*dHHY,*dx;
    cudaMalloc((void **)&dH, K*N*sizeof(cuFloatComplex));
    cudaMalloc((void **)&dHH,  N*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dmHH, N*N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dInv, N*N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dInvH, N*N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dInvM, N*N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dY, K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dHHY, K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dx, K*sizeof(cuFloatComplex));


	cudaEventRecord(start2, 0);
    //Copy input data to array on GPU.
    cudaMemcpy(dH, H, K*N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, K*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

	//Run the transpose on gpu
	//Number of threads are K*N with k-rows and N-columns
	dim3 blockDims(K,N);
	dim3 GridDims(1);
	
	cudaEventRecord(start, 0);
    hermitian_transpose<<<blockDims,GridDims>>>(dH, dHH,K,N);
	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time for only H^H: %f ms.\n", elapsed_time_ms);
	
	
	blockDims.x = N;
	
	//Number of threads are N*N
	cudaEventRecord(start, 0);
	Ltriangle_complex_matrix_mult<<<blockDims,GridDims>>>(dHH, dH, dmHH,N,K,N);
	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time for only H*H^H (lower triangle): %f ms.\n", elapsed_time_ms);
	cudaDeviceSynchronize();

/* the original
	for(int i = 0; i < N; i++){
		//Del1 of cholesky. (Diagonal element) one thread
		bChol<<<1,1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//Part2 of cholesky (column compleeted)
		//the amount of threads is getting smaler each itteration
		//it is the number of elements in the vector under the diagonal element
		bChol2<<<N-(i+1),1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//Part3 of cholesky and start cInv part1
		cInv1<<<i+1,1>>>(dmHH,dInv,i,N);
		blockDims.x = N-(i+1);
		blockDims.y = N-(i+1);
		bChol3<<<blockDims,1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//Part2 of inv
		blockDims.x = N-(i+1);
		blockDims.y = N;
		cInv2<<<blockDims,1>>>(dmHH,dInv,i,N);
		//printf("\n");
	}
*/	
	
	cudaEventRecord(start, 0);
	for(int i = 0; i < N; i++){
		//Del1 of cholesky. (Diagonal element) one thread
		bChol<<<1,1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//Part2 of cholesky (column compleeted)
		bChol2<<<N-(i+1),1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//Part3 of cholesky and start cInv part1
		blockDims.x = N-(i+1);
		blockDims.y = N-(i+1);
		bChol3<<<blockDims,1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//printf("\n");
	}
	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time for only cholesky: %f ms.\n", elapsed_time_ms);
	
	cudaEventRecord(start, 0);
	for(int i = 0; i < N; i++){
		//Part1 cInv
		cInv1<<<i+1,1>>>(dmHH,dInv,i,N);
		cudaDeviceSynchronize();
		//Part2 of inv
		blockDims.x = N-(i+1);
		blockDims.y = N;
		cInv2<<<blockDims,1>>>(dmHH,dInv,i,N);
		cudaDeviceSynchronize();
	}
	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time for only inversion: %f ms.\n", elapsed_time_ms);
	
	//This part takes the inv of L multiplied with itsef to become A^-1
	blockDims.x = N;
	blockDims.y = N;
    hermitian_transpose<<<blockDims,GridDims>>>(dInv, dInvH,N,N);
	cudaDeviceSynchronize();
	//complex_matrix_mult<<<blockDims,GridDims>>>(dInv, dInvH, dInvM,K,K,K); Right way but not for Ali
	complex_matrix_mult<<<blockDims,GridDims>>>(dInvH, dInv, dInvM,N,N,N);
	
	blockDims.x = N;
	blockDims.y = 1;
	//dHH = 8x128 dy = 128x1 dHHY = 8x1
	cudaEventRecord(start, 0);
	complex_matrix_mult<<<blockDims,GridDims>>>(dHH, dY, dHHY,N,K,1);
	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time for only H^H*y: %f ms.\n", elapsed_time_ms);
	cudaDeviceSynchronize();
	//dHH = 8x8 dHHY = 8x1
	cudaEventRecord(start, 0);
	complex_matrix_mult<<<blockDims,GridDims>>>(dInvM, dHHY, dx,N,N,1);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time for only A^-1*B: %f ms.\n", elapsed_time_ms);
	
	cudaMemcpy(hHHY, dx, K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop2, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&elapsed_time_ms2, start2, stop2);
	printf("Time for everything: %f ms.\n", elapsed_time_ms2);
	
	cudaMemcpy(hmHH, dInv, N*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);


	printf("X_zf = ;\n");
	for (int i = 0; i<N; ++i) {
		for (int j = 0; j<1; ++j) {
			printf("%f+%fi ", hHHY[j*K+i].x,hHHY[j*K+i].y);
		}
        printf(";\n");
    }
    
	//Free from CPU
	free(hY);
	free(H);
    // Free up the arrays on the GPU.
    cudaFree(dH);
    cudaFree(dHH);
	cudaFree(dmHH);
	cudaFree(dInv);
	cudaFree(dInvH);
	cudaFree(dInvM);
	cudaFree(dY);
	cudaFree(dHHY);

    return 0;
}