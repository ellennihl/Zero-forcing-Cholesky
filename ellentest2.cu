#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>


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

__global__ void complex_matrix_mult2(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {

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


__global__ void row_by_row_complex_matrix_mult(const float2 *L, float2 *A_inv, const int size, const int current_row){
	
	int row = threadIdx.x + blockDim.x * blockIdx.x;
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	
	if (row < size && col < size && (row == current_row || col == current_row)) {	
		float2 sum = make_float2(0.0f, 0.0f);
		
		printf("index rowbyrowmatmul: (row,col): (%d,%d)\n", row, col);
		for(int k = 0; k < size; k++){
			/*
			this works for L*LH
			float2 a = L[k * size + row]; 
            float2 b = L[k * size + col];

			b = make_float2(b.x, -b.y);//conjugate
			*/
			float2 b = L[k * size + row]; 
            float2 a = L[k * size + col];

			a = make_float2(a.x, -a.y);//conjugate
			//float2 a = A[k * res_row + row]; //column-major!!!!!!
            //float2 b = B[col * a_col_b_row + k];
			
			//float2 b = L[k * size + current_row];
            //float2 a = L[k * size + col];
			//a = make_float2(a.x, -a.y);//conjugate
			

			
			printf("(row,col): (%d,%d) idx LH: %d, idx L: %d a: %f %fi b: %f %fi test: %f %fi\n",row,col, k * size + col, k * size + current_row,a.x,a.y,b.x,b.y,L[col * size + k].x,L[col * size + k].y);
			
			float real_part = a.x * b.x - a.y * b.y;
			float imag_part = a.x * b.y + a.y * b.x;

			sum.x += real_part;
			sum.y += imag_part;	
		}
		
		A_inv[col * size + row] = sum;
		printf("(row,col): (%d,%d) sum: %f %fi\n",row,col,sum.x,sum.y);
	}
}


int main() {

	//Size of matrix N=antennas, K=Users
	
	int N = 3;
	int K = 3;
	
	cuFloatComplex hY[N];
	//initializing y matrix
	hY[0].x = -1.15044381816198;
	hY[0].y = 2.80297100338098;
	hY[1].x = -1.45737148064847;
	hY[1].y = 0.105134117295914;
	hY[2].x = -2.73160735027786;
	hY[2].y = -0.0430050084558768;
	
	//The h stands for host
	cuFloatComplex H[N*K],hHH[K*K], hmHH[K*K],hInv[K*K],hHHY[N],hInv2[K*K];
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
	
	//The d stands for device
    cuFloatComplex *dH, *dHH, *dmHH, *dInv, *dInvH,*dInvM,*dY,*dHHY,*dx;
    cudaMalloc((void **)&dH, N*K*sizeof(cuFloatComplex));
    cudaMalloc((void **)&dHH,  K*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dmHH, K*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dInv, K*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dInvH, K*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dInvM, K*K*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dY, N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dHHY, N*sizeof(cuFloatComplex));
	cudaMalloc((void **)&dx, N*sizeof(cuFloatComplex));
	
	cudaEvent_t start, stop;     		// using cuda events to measure time
	float elapsed_time_ms;       		// which is applicable for asynchronous code also

	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);
//	cudaEventSynchronize(start);  	// Needed?

    //Copy input data to array on GPU.
    cudaMemcpy(dH, H, N*K*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
	cudaMemcpy(dY, hY, N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);

	//Run the transpose on gpu
	dim3 blockDims(N,K);
	dim3 GridDims(1);
	dim3 blockDimsMult(1,1);
    hermitian_transpose<<<blockDims,GridDims>>>(dH, dHH,N,K);
	//Run the multiplication on the GPU
	blockDims.x = K;
	complex_matrix_mult2<<<blockDims,GridDims>>>(dHH, dH, dmHH,N,K,N);
	cudaDeviceSynchronize();
	//Run the cholesky algorithem
	for(int i = 0; i < K; i++){
		//Del1 of cholesky. (Diagonal element)
		bChol<<<1,1>>>(dmHH,i,K);
		cudaDeviceSynchronize();
		//Part2 of cholesky (column compleeted)
		bChol2<<<K-(i+1),1>>>(dmHH,i,K);
		cudaDeviceSynchronize();
		//Part3 of cholesky and start cInv part1
		cInv1<<<i+1,1>>>(dmHH,dInv,i,K);
		blockDims.x = K-(i+1);
		blockDims.y = K-(i+1);
		bChol3<<<blockDims,1>>>(dmHH,i,K);
		cudaDeviceSynchronize();
		//Part2 of inv
		blockDims.x = K-(i+1);
		blockDims.y = K;
		cInv2<<<blockDims,1>>>(dmHH,dInv,i,K);
		blockDimsMult.x = i+1;
		blockDimsMult.y = i+1;
		row_by_row_complex_matrix_mult<<<blockDimsMult, 1>>>(dInv,dInvM,K,i);//testing
		printf("\n");
	}
	
	//This part takes the inv of L multiplied with itsef to become A^-1
	blockDims.x = K;
	blockDims.y = K;
//ellen test att byta ut
/*    hermitian_transpose<<<blockDims,GridDims>>>(dInv, dInvH,K,K);
	cudaDeviceSynchronize();
	//complex_matrix_mult<<<blockDims,GridDims>>>(dInv, dInvH, dInvM,K,K,K); Right way but not for Ali
	complex_matrix_mult<<<blockDims,GridDims>>>(dInvH, dInv, dInvM,K,K,K);
	*/
	cudaDeviceSynchronize();
	complex_matrix_mult<<<blockDims,GridDims>>>(dHH, dY, dHHY,N,K,1);
	cudaDeviceSynchronize();
	complex_matrix_mult<<<blockDims,GridDims>>>(dInvM, dHHY, dx,N,K,1);

	cudaMemcpy(hHHY, dx, N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);     	// instrument code to measue end time
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop );
	printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);


	//Coppy the results to the host
    cudaMemcpy(hHH, dHH, N*K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    cudaMemcpy(hmHH, dmHH, K*K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(hInv, dInvM, K*K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
	cudaMemcpy(hInv2, dInv, K*K*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);

	printf("H^HH = \n");
	//Print out the gramian matrix
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<K; ++j) {
			printf("%f+%fi ", hHH[j*K+i].x,hHH[j*K+i].y);
		}
        printf(";\n");
    }
	
	printf("L = \n");
	//Print out the cholesky matrix
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<K; ++j) {
			printf("%f+%fi ", hmHH[j*K+i].x,hmHH[j*K+i].y);
		}
        printf(";\n");
    }

	printf("Inv no mult= \n");
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<K; ++j) {
			printf("%f+%fi ", hInv2[j*K+i].x,hInv2[j*K+i].y);
		}
        printf(";\n");
    }
	
	printf("Inv and mult= \n");
	for (int i = 0; i<K; ++i) {
		for (int j = 0; j<K; ++j) {
			printf("%f+%fi ", hInv[j*K+i].x,hInv[j*K+i].y);
		}
        printf(";\n");
    }
	
	printf("x = \n");
	for (int i = 0; i<N; i++) {
		printf("%f+%fi ", hHHY[i].x,hHHY[i].y);
        printf(";\n");
    }
	
    // Free up the arrays on the GPU.
    cudaFree(dH);
    cudaFree(dHH);
	cudaFree(dmHH);
	cudaFree(dInv);
	cudaFree(dInvH);
	cudaFree(dInvM);
	cudaFree(dY);
	cudaFree(dHHY);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    return 0;
}