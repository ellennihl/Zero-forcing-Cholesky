//serial execution
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <complex.h>

typedef struct float2{
	float x; //real
	float y; //imaginary
} float2;

/**
	Takes a csv containing a matrix and returns an array column major
*/
float2 *read_matrix_from_csv(char filename[], int num_rows, int num_cols) {
	// Allocate memory for the matrix
	float2 *matrix = (float2 *) malloc(num_rows * num_cols * sizeof(float2));
	int real = 1;

	char tempchar[20];
	strcpy(tempchar, filename);
	for(int i=0;i<2;i++){
		if(i == 0){
			strcat(filename, "_real.csv");
			real = 1;
		}
		else{
			strcat(tempchar, "_imag.csv");
			strcpy(filename, tempchar);
			real = 0;
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


void hermitian_transpose(const float2* input_h, float2* output_hh, int K, int N) { //const because we do not want to modify the input matrix
	
	for(int row = 0; row<N; row++){
		for(int col = 0; col<K; col++){

			int idx_in = col * N + row;
			int idx_out = row * K + col;
			
			output_hh[idx_out].x = input_h[idx_in].x; //conjugate
			output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part

		}
	}
}

void complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {

	for(int row = 0; row<res_row; row++){
		for(int col = 0; col<res_col; col++){
        float2 sum = {.x = 0.0f, .y = 0.0f};

        for (int k = 0; k < a_col_b_row; k++) {
			//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);

            float2 a = A[k * res_row + row]; //column-major!!!!!!
            float2 b = B[col * a_col_b_row + k];
            float real_part = a.x * b.x - a.y * b.y;
            float imag_part = a.x * b.y + a.y * b.x;
            sum.x += real_part;
            sum.y += imag_part;
			}
			
		C[col * res_row + row] = sum;
		}
	}
}

void L_triangular_complex_matrix_mult(const float2* A, const float2* B, float2* C, const int res_row, const int a_col_b_row, const int res_col) {

	for(int row = 0; row<res_row; row++){
		for(int col = 0; col<res_col; col++){
			float2 sum = {.x = 0.0f, .y = 0.0f};
			if(row>=col){
				for (int k = 0; k < a_col_b_row; k++) {
					//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);

					float2 a = A[k * res_row + row]; //column-major!!!!!!
					float2 b = B[col * a_col_b_row + k];
					float real_part = a.x * b.x - a.y * b.y;
					float imag_part = a.x * b.y + a.y * b.x;
					sum.x += real_part;
					sum.y += imag_part;
					}
			}
			
		C[col * res_row + row] = sum;
		}
	}
}

float2 csqrt(float2 z){
	float complex z = z.x + z.y*I;
	float r = cabsf(z);
    float theta = atan2(z.y,z.x);
    float2 sqrt_z = {.x = sqrtf(r) * cosf(theta / 2.0f), .y = sqrtf(r) * sinf(theta / 2.0f)}//;make_cuFloatComplex(sqrtf(r) * cosf(theta / 2.0f),
                                                //sqrtf(r) * sinf(theta / 2.0f));
	return sqrt_z;
}

void bChol(float2* A,int i,int N){
	A[i*N+i] = csqrt(A[i*N+i]);
}

void bChol2(float2* A,int i,int N){
	//int row = blockIdx.x + 1;
	//int diagonal = i*N+i;
	/*if(row == 1){
		//printf("sqtr %d \n",i*N+i); this is a 
		A[i*N+i] = cuCsqrt(A[i*N+i]);
	}*/
	//__syncthreads();
	//printf("\n %d %d",i*N+i, (i*N+i)+row);
	int range = N-(i+1);
	for(int row = i+1; row<range; row++){
		A[(i*N+i)+row] = (A[(i*N+i)+row].x*A[i*N+i].x + A[(i*N+i)+row].y * A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y);//cuCdivf(A[(i*N+i)+row], A[i*N+i]);
	}
}

void bChol3(float2* A, int i, int N){
	int j = i+1;
	int range = N-(i+1);
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	//int vector = threadIdx.y + blockDim.y * blockIdx.y;
	for(int row = 0; row < range; row++){
		for(int col = 0; col < range; col++){
			if(row >= col){
				//printf("vec1 %d vec2 %d (%d,%d) index %d \n",(N*i+i+1)+row,(N*i+i+1)+col,row,col,(col+j)*N+j+row);
				//printf("%d = %d-%d*%d \n",(col+j)*N+j+row,(col+j)*N+j+row,(N*i+i+1)+row,(N*i+i+1)+col);
				float2 tmp = A[(N*i+i+1)+col];
				tmp.y = -tmp.y;
				A[(col+j)*N+j+row] = cuCsubf(A[(col+j)*N+j+row],cuCmulf(A[(N*i+i+1)+row],tmp));
			}
			
		}
		
	}
	
}

int main() {

	//read the Y.csv
	int num_rows = 128;
	int num_cols = 1;
	float2 *Y;
	char file[] = "128x8/Y";
	Y = read_matrix_from_csv(file, num_rows, num_cols);

	//read H.csv
	num_rows = 128;
	num_cols = 8;
	float2 *H;
	strcpy(file, "128x8/H");
	H = read_matrix_from_csv(file, num_rows, num_cols);

	//Size of N=antennas (nr of rows), K=Users (nr of columns)
	int N = 128;
	int K = 8;
	
	/*printf("---------------H---------------------------\n");
	for(int i=0; i<N; i++){//rows
		for(int j = 0; j<K; j++){//cols
			printf("%f+%fi	",H[j*N + i].x,H[j*N + i].y);//N*K
		}
		printf("\n");
	}
	printf("-------------------------------------------\n");
*/
/*
	float2 inv[K*K];
	float2 HHY[N];
	*/
	float2 *HH, *mHH, *dInv, *dInvH,*dInvM,*dY,*HHY,*dx;
	HH = (float2 *) malloc(K * N * sizeof(float2));
	mHH = (float2 *) malloc(K * K * sizeof(float2));
	HHY = (float2 *) malloc(N * 1 * sizeof(float2));
	//mHH = (float2 *) malloc(K * K * sizeof(float2));
	
	hermitian_transpose(H, HH, K, N);
	
	printf("hermitian transpose------------------------------\n");
	for(int i=0; i<K; i++){//rows
		for(int j = 0; j<N; j++){//cols
			printf("%f+%fi	",HH[j*K + i].x,HH[j*K + i].y);//K*N
		}
		printf(" ; \n");
	}
	printf("------------------------------------------\n");
	
	L_triangular_complex_matrix_mult(HH, H, mHH, K, N, K);
	
	printf("matmul------------------------------------------\n");
	for(int i=0; i<K; i++){//rows
		for(int j = 0; j<K; j++){//cols
			printf("%f+%fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*K
		}
		printf(" ; \n");
	}
	printf("------------------------------------------\n");
	
	complex_matrix_mult(HH, Y, HHY, K, N, 1);
	
	printf("matvecmul------------------------------------------\n");
	for(int i=0; i<K; i++){//rows
		for(int j = 0; j<1; j++){//cols
			printf("%f+%fi	",HHY[j*K + i].x,HHY[j*K + i].y);//K*1
		}
		printf(" ; \n");
	}
	printf("------------------------------------------\n");
	
	for(int i = 0; i < K; i++){
		//Del1 of cholesky. (Diagonal element) one thread
		bChol(mHH,i,K);
		bChol2(mHH,i,K);
		//bChol<<<1,1>>>(mHH,i,N);
		//cudaDeviceSynchronize();
		//Part2 of cholesky (column compleeted)
		//the amount of threads is getting smaler each itteration
		//it is the number of elements in the vector under the diagonal element
		//bChol2<<<K-(i+1),1>>>(dmHH,i,N);
		//cudaDeviceSynchronize();
		//Part3 of cholesky and start cInv part1
	/*	cInv1<<<i+1,1>>>(dmHH,dInv,i,N);
		blockDims.x = N-(i+1);
		blockDims.y = N-(i+1);
		bChol3<<<blockDims,1>>>(dmHH,i,N);
		cudaDeviceSynchronize();
		//Part2 of inv
		blockDims.x = N-(i+1);
		blockDims.y = N;
		cInv2<<<blockDims,1>>>(dmHH,dInv,i,N);
		//printf("\n");*/
	}
	
}