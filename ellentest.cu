// Ellen test
//compile when makefile is being weird: 
///usr/local/cuda/bin/nvcc -I/usr/local/cuda/include -L /usr/local/cuda/lib64 -lcuda -lcudart -lm -o EllenTest ellentest.cu

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <cuComplex.h>

//float2 and cuFloatComplex are the same thing, 
//in cuComplex.h: typedef float2 cuFloatComplex
__device__ cuFloatComplex cuCsqrt(cuFloatComplex z){
	float r = cuCabsf(z);
    float theta = atan2(z.y,z.x);
    cuFloatComplex sqrt_z = make_cuFloatComplex(sqrtf(r) * cosf(theta / 2.0f),
                                                sqrtf(r) * sinf(theta / 2.0f));
	return sqrt_z;
}

__global__ void hermitian_transpose(const float2* input_h, float2* output_hh, const int N, const int K) { //const because we do not want to modify the input matrix!!!

	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3

    if (col < K && row < N) {
		
		int idx_in = col * N + row; //index in matrix
		int idx_out = row * K + col; //output should be reversed (transpose)

		//conjugate here - in a float2: .x is the real part, .y is imaginary part
        output_hh[idx_out].x = input_h[idx_in].x; //conjugate
        output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
    }
}

//axb * cxd = axd
//b=c otherwise matmul cant happen
//K*M * N*1 = K*1
__global__ void complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int b_row_a_col, const int res_col) {

	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	if (row < res_row && col < res_col) {
        float2 sum = make_float2(0.0f, 0.0f);

        for (int k = 0; k < b_row_a_col; k++) {
			float2 a = A[k * res_row + row]; //column-major!!!!!!
            float2 b = B[col * b_row_a_col + k];
			
			//printf("(%d,%d) a: %d   b: %d\n",row,col, k * res_row + row, k * res_col + col);
			
            float real_part = a.x * b.x - a.y * b.y;
            float imag_part = a.x * b.y + a.y * b.x;

            sum.x += real_part;
            sum.y += imag_part;
        }

		C[col * res_row + row] = sum;
		//printf("(%d,%d) result index: %d\n",row,col, col * res_row + row);
		//if column done (col == K)- set event for cholesky?
	}
}

//column by column
__global__ void cholesky(float2 *A, int column, const int size, float2 *L_T){
	
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	int idx = column * size + row + column; //find index
	int transpose_idx = row * size + column; //WRONG!!!!!!!!!!!!!!!!!!!
	int diagonal = (column * size) + column; //get diagonal element index
	
	if(idx == diagonal){//part 1, if diagonal element
	
		cuFloatComplex sq = cuCsqrt(A[idx]);
		printf("sqrt: %f %f\n", sq.x, sq.y);
		A[idx] = sq;
		L_T[transpose_idx] = A[idx];//MÅSTE FÖRMODLIGEN BYTA TECKEN
	
	}
	__syncthreads(); //every thread needs to reach this place before continuing execution
	if(idx != diagonal){//part 2
	
		A[idx] = cuCdivf(A[idx], A[diagonal]);//A[idx]/A[diagonal]
		L_T[transpose_idx] = A[idx];
	
	}
	printf("chol: (%d,%d) idx: %d column: %d diagonal: %d \n", row,col,idx,column, diagonal);
}

//block size has to be of size of U: (K-column)x(K-column)
__global__ void cholesky_part3(float2 *A, const int column, const int K){
	
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	
	int U_start_idx = (column+1) * K + (column+1); 	//start index of U 
	int U_iterate_idx = col * K + row;				//iterate through U
	int U_idx = U_start_idx + U_iterate_idx;		//U index for this thread
	
	//U-v1*v2, where index of U tells what index of v to multiply
	//example: U(0,0) = v(0)*v(0), and U(1,2)=v(1)*v(2)
	int vector_start_idx = column*K+column+1; 	//vector starts at the current column (which is column*K)
												//and then go past the elements over U (which is column amount) 
												//and then +1 because skip diagonal element
	int vec1_idx = vector_start_idx + row; 	//first vector index, corresponding to U's row
	int vec2_idx = vector_start_idx + col;	//same but column index instead
	
	float2 vec1_star = make_float2(A[vec1_idx].x, -A[vec1_idx].y); //L*!!!!! for complex numbers
	
	A[U_idx] = cuCsubf(A[U_idx],cuCmulf(vec1_star,A[vec2_idx]));//A[U_idx] = A[U_idx] - A[vec1_idx]*A[vec2_idx] but with complex nrs
	printf("in part3:\n(%d,%d): U_idx: %d  vec1_idx: %d  vec2_idx: %d, A[U_idx]: %f %fi, A[vec1_idx]: %f %fi, A[vec2_idx]: %f %fi\n",row,col, U_idx, vec1_idx, vec2_idx, A[U_idx].x, A[U_idx].y, A[vec1_idx].x, A[vec1_idx].y, A[vec2_idx].x, A[vec2_idx].y);
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
		printf("P2 (%d,%d) %d\n",row+i+1,col,i);
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


__global__ void row_by_row_complex_matrix_mult(const float2 *L, float2 *A_inv, const int size, const int current_row){
	
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	
	//int row_start_idx = current_row;
	//int row_iteration = size;
	
	//0,1,2
	//int A_row = current_row*row;
	//int A_col = current_row*col;
	
	if (row < size && col < size && (row == 1 || col == 1)) {	
		float2 sum = make_float2(0.0f, 0.0f);
		//int L_idx = current_row;
		//int L_T_idx = 
		
		for(int k = 0; k < size; k++){
			float2 a = L[k * size + current_row]; //column-major!!!!!!
            float2 b = L[k * size + col];
			
			//float2 a = L[k * size + row]; 
			//float2 b = L[k * size + row];//L[col * size + k]; //this is L_T
			b = make_float2(b.x, -b.y);//conjugate
							
			float real_part = a.x * b.x - a.y * b.y;
			float imag_part = a.x * b.y + a.y * b.x;

			sum.x += real_part;
			sum.y += imag_part;	
		}
		
		A_inv[col * size + row] = sum;
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

	//float2 h[N*K] = { {1.0f, 2.0f}, {5.0f, 6.0f}, {9.0f, 10.0f}, {13.0f, 14.0f}, {3.0f, 4.0f}, {7.0f, 8.0f}, {11.0f, 12.0f}, {15.0f, 16.0f} };
	float2 h[N*K] = { {1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}, {13.0f, 14.0f}, {15.0f, 16.0f} };
	float2 y[N] =  { {1.0f, 3.0f}, {4.0f, 8.0f}, {16.0f, 8.0f}, {2.0f, 2.0f} };
	
	for (int i = 0; i < N*K; i++) { //print input matrix
		printf("(%f + %fi)\n", h[i].x, h[i].y);
	}
	
	float2 *mat_h,*mat_hh,*mat_hhh, *vec_y, *vec_hy, *mat_l, *mat_lh; //float2 z = {1.0f, 2.0f}; // z = 1.0 + 2.0i
	int matKN_size;					// number of bytes in arrays
	int matKK_size;
	int vecN_size;
	int vecK_size;
	
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

	dim3 Grid(Grid_Dim_x, Grid_Dim_y);		//Grid structure
	dim3 Block(Block_Dim_x,Block_Dim_y);	//Block structure, threads/block limited by specific device

	matKN_size = K * N * sizeof(float2); //Hh and H are K*N and N*K
	matKK_size = K * K * sizeof(float2);	//gramian is K*K
	vecN_size = N * sizeof(float2);		//vec is K*1
	vecK_size = K * sizeof(float2);

/* ------------- COMPUTATION DONE ON GPU ----------------------------*/

	cudaMalloc((void**)&mat_h, matKN_size);		// allocate memory on device
	cudaMalloc((void**)&mat_hh, matKN_size);
	cudaMalloc((void**)&mat_hhh, matKK_size);
	cudaMalloc((void**)&vec_y, vecN_size); //size of y vector is Nx1
	cudaMalloc((void**)&vec_hy, vecK_size); //size of Hy vector is Nx1
	cudaMalloc((void**)&mat_l, matKK_size);
	cudaMalloc((void**)&mat_lh, matKK_size);

	cudaMemcpy(mat_h, h, matKN_size ,cudaMemcpyHostToDevice); //put h in device
	cudaMemcpy(vec_y, y, vecN_size ,cudaMemcpyHostToDevice); //put y in device

//--------------------------TRANSPOSE-Hh---------------------------------
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	hermitian_transpose<<<Grid,Block>>>(mat_h,mat_hh,N,K); //calc hermitian Hh

	float2 output[N*K];//just to print, device has mat_hh, host does not need it?
	cudaMemcpy(output, mat_hh, matKN_size, cudaMemcpyDeviceToHost);

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
	complex_matrix_mult<<<Grid,Block>>>(mat_hh, mat_h, mat_hhh, K,N,K);

	float2 gramian[K*K];
	cudaMemcpy(gramian, mat_hhh, matKK_size, cudaMemcpyDeviceToHost);

	printf("gramian\n");
	for (int i = 0; i < K*K; i++) {
		printf("(%f + %fi)\n", gramian[i].x, gramian[i].y);
	}
	
//-------------------------MAT-VEC-MUL-Hy--------------------------------------
	//this is Hhy
	complex_matrix_mult<<<Grid,Block>>>(mat_hh, vec_y, vec_hy, K,N,1); 
	
	float2 hy[K];
	cudaMemcpy(hy, vec_hy, vecK_size, cudaMemcpyDeviceToHost);
	
	printf("Hy\n");
	for (int i = 0; i < K; i++) {
		printf("(%f + %fi)\n", hy[i].x, hy[i].y);
	}

//--------------------------Cholesky--------------------------------------
//this should be in a thread later
	Grid_Dim_x = 1;
	Grid_Dim_y = 1;
	Block_Dim_x = 2;
	Block_Dim_y = 1;
	
	int part3_grid_dim_x = 1;
	int part3_grid_dim_y = 1;
	int part3_block_dim_x = K-1;//size of U is (column-1)x(column-1)
	int part3_block_dim_y = K-1;
	
    for (int col = 0; col < K; col++) {
		
		dim3 Grid2(Grid_Dim_x, Grid_Dim_y);		//Grid structure
		dim3 Block2(Block_Dim_x--,Block_Dim_y);	//Block structure, threads/block limited by specific device
		
		//launch cholesky part 1 and 2
        cholesky<<<Grid2, Block2>>>(mat_hhh, col, K, mat_lh);

        //if (col > 0) {
        cudaDeviceSynchronize();//now one column is finished
        //}
		if(col < K-1){
			dim3 Grid_part3(part3_grid_dim_x,part3_grid_dim_y);
			dim3 Block_part3(part3_block_dim_x--, part3_block_dim_y--);
			//launch part 3
			cholesky_part3<<<Grid2,Block_part3>>>(mat_hhh, col, K);
			cudaDeviceSynchronize();

		}
    }
	//testar innan detta bara 
	//dim3 Grid(Grid_Dim_x, Grid_Dim_y);		//Grid structure
	//dim3 Block(Block_Dim_x,Block_Dim_y);	//Block structure, threads/block limited by specific device

	//matKN_size = K * N * sizeof(float2);
	//row_by_row_complex_matrix_mult<<<Grid,Block>>>(mat_hhh, );
	
	float2 l[K*K];
	
	cudaMemcpy(l, mat_hhh, matKK_size, cudaMemcpyDeviceToHost);
	
	printf("lite chol\n");
	for(int i = 0; i < K*K; i++){
		printf("%f + %f\n",l[i].x, l[i].y);
		
	}

/* --------------  clean up  ---------------------------------------*/
	cudaFree(mat_h);
	cudaFree(mat_hh);
	cudaFree(mat_hhh);
	cudaFree(vec_y);
	cudaFree(vec_hy);
	cudaFree(mat_l);
	cudaFree(mat_lh);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return 0;
}

void AliTest(float2 *y, float2 *H){
	
	y[0].x = -1.15044381816198;
	y[0].y = 2.80297100338098;
	y[1].x = -1.45737148064847;
	y[1].y = 0.105134117295914;
	y[2].x = -2.73160735027786;
	y[2].y = -0.0430050084558768;
	
	//initializing H matrix
	H[0].x = -0.14871528137562;
	H[0].y = -0.839585070157793;
	H[1].x = 0.456796194001739;
	H[1].y = -1.39648740223667;
	H[2].x = -0.627350895700304;
	H[2].y = -0.491338636279611;
	H[3].x = 0.756444232794338;
	H[3].y = -0.238637048003854;
	H[4].x = -0.374235630126775;
	H[4].y = 0.686050058020553;
	H[5].x = 0.959923600007699;
	H[5].y = -0.0923017429928966;
	H[6].x = 1.25391777895517;
	H[6].y = 0.0860634779712874;
	H[7].x = -0.322123665443045;
	H[7].y = -0.101934261054657;
	H[8].x = -0.727806592386333;
	H[8].y = 0.0283459633648643;
	
}