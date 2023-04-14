#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
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

/*
	0	0	0
	0	0	0
	0	0	0
*/

//This part can be parallelised
//Step3 U-c*c^T
__global__ void bChol3(float2* A, int i, int N){
	int j = i+1;
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	int vector = threadIdx.y + blockDim.y * blockIdx.y;
	if(row >= col){
		printf("vec1 %d vec2 %d (%d,%d) index %d \n",(N*i+i+1)+row,(N*i+i+1)+col,row,col,(col+j)*N+j+row);
		printf("%d = %d-%d*%d \n",(col+j)*N+j+row,(col+j)*N+j+row,(N*i+i+1)+row,(N*i+i+1)+col);
		float2 tmp = A[(N*i+i+1)+col];
		tmp.y = -tmp.y;
		A[(col+j)*N+j+row] = cuCsubf(A[(col+j)*N+j+row],cuCmulf(A[(N*i+i+1)+row],tmp));
	}
}

//float2 and cuFloatComplex are the same thing, 
//in cuComplex.h: typedef float2 cuFloatComplex
__global__ void bChol2(float2* A,int i,int N){
	int row = blockIdx.x + 1;
	//int diagonal = i*N+i;
	if(row == 1){
		//printf("sqtr %d \n",i*N+i); this is a 
		A[i*N+i] = cuCsqrt(A[i*N+i]);
	}
	__syncthreads();
	
	printf("\n %d %d",i*N+i, (i*N+i)+row);
    A[(i*N+i)+row] = cuCdivf(A[(i*N+i)+row], A[i*N+i]);
}

/**	Step 1 sqrt(A_ii);
	A is the matrix that is decomposed
	i is the column of the matrix that is being decomopesed
	N is the size of the N*N matrix
*/
__global__ void bChol(float2* A,int i,int N) {
    A[i*(N+1)] = cuCsqrt(A[i*(N+1)]);
}

/**
	This metod takes in a matrix and returns the hermetian transpose of the matrix	
	input_h is the input matrix with size NxK
	output_hh is the resulting matrix with size KxN
	N is the nr of in input_h
	K the nr of rows in input_h
*/
__global__ void hermitian_transpose(const float2* input_h, float2* output_hh, int N, int K) { //const because we do not want to modify the input matrix!!!
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	if (col < K && row < N) {
		int idx_in = col * N + row;
		int idx_out = row * K + col;
		//printf("(%d,%d)  in: %d, out: %d\n",row,col,idx_in,idx_out);
		//conjugate here - in a float2: .x is the real part, .y is imaginary part
        output_hh[idx_out].x = input_h[idx_in].x; //conjugate
        output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
    }
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
		if(row >= col){
			C[col * res_row + row] = sum;
		}
	}
}

int main() {

	//Size of matrix N=antennas, K=Users
	//
	int N = 3;
	int K = 3;
	
	cuFloatComplex y[N];
	//initializing y matrix
	y[0].x = -1.15044381816198;
	y[0].y = 2.80297100338098;
	y[1].x = -1.45737148064847;
	y[1].y = 0.105134117295914;
	y[2].x = -2.73160735027786;
	y[2].y = -0.0430050084558768;
	
	//The h stands for host
	cuFloatComplex H[N*K],hHH[K*K], hmHH[K*K];
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
	/*
	H[0].x = 2;
	H[1].x = 2;
	H[2].x = 3;
	H[3].x = 2;
	H[4].x = 10;
	H[5].x = 5;
	H[6].x = 3;
	H[7].x = 5;
	H[8].x = 20;
	H[0].y = 0;
	H[1].y = 0;
	H[2].y = 0;
	H[3].y = 0;
	H[4].y = 0;
	H[5].y = 0;
	H[6].y = 0;
	H[7].y = 0;
	H[8].y = 0;
	*/
	//The d stands for device
    cuFloatComplex *dH, *dHH, *dmHH;
    cudaMalloc((void **)&dH, N*K*sizeof(cuFloatComplex));
    cudaMalloc((void **)&dHH, K*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dmHH, K*K*sizeof(cuFloatComplex));

    //Copy input data to array on GPU.
    cudaMemcpy(dH, H, N*K*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

	//Run the transpose on gpu
	dim3 blockDims(N,K);
	dim3 GridDims(1);
    hermitian_transpose<<<blockDims,GridDims>>>(dH, dHH,N,K);
	//Run the multiplication on the GPU
	blockDims.x = K;
	complex_matrix_mult<<<blockDims,GridDims>>>(dHH, dH, dmHH,N,K,N);
	cudaDeviceSynchronize();
	//Run the cholesky algorithem
	for(int i = 0; i < K; i++){
		bChol2<<<K-(i+1),1>>>(dmHH,i,K);
		if(i==K-1){bChol2<<<1,1>>>(dmHH,i,K);}
		printf("\n");
		cudaDeviceSynchronize();
		blockDims.x = K-(i+1);
		blockDims.y = K-(i+1);
		bChol3<<<blockDims,1>>>(dmHH,i,K);
		printf("\n");
		cudaDeviceSynchronize();
	}

	//Coppy the results to the host
    cudaMemcpy(hHH, dHH, N*K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(hmHH, dmHH, K*K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

    /*
	//Test for hermetian transpose
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<K; ++j) {
			printf("%d", H[i*K+j].x == hHH[j*K+i].x);
			printf("%d ", H[i*K+j].y == -hHH[j*K+i].y);
		}
        printf("\n");
    }
	*/
	printf("\n");
	
	
	//Print out the gramian matrix
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<K; ++j) {
			printf("%f+%fi ", hmHH[j*K+i].x,hmHH[j*K+i].y);
		}
        printf(";\n");
    }
	
    // Free up the arrays on the GPU.
    cudaFree(dH);
    cudaFree(dHH);
	cudaFree(dmHH);
    return 0;
}