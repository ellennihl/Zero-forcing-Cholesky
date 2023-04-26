//serial execution
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <complex.h>

typedef struct double2{
	double x; //real
	double y; //imaginary
} double2;

/**
	Takes a csv containing a matrix and returns an array column major
*/
double2 *read_matrix_from_csv(char filename[], int num_rows, int num_cols) {
	// Allocate memory for the matrix
	double2 *matrix = (double2 *) malloc(num_rows * num_cols * sizeof(double2));
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


void hermitian_transpose(const double2* input_h, double2* output_hh, int K, int N) { //const because we do not want to modify the input matrix
	
	for(int row = 0; row<N; row++){
		for(int col = 0; col<K; col++){

			int idx_in = col * N + row;
			int idx_out = row * K + col;
			
			output_hh[idx_out].x = input_h[idx_in].x; //conjugate
			output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part

		}
	}
}

void complex_matrix_mult(const double2* A, const double2* B, double2* C, const int res_row, const int a_col_b_row, const int res_col) {

	for(int row = 0; row<res_row; row++){
		for(int col = 0; col<res_col; col++){
        double2 sum = {.x = 0.0f, .y = 0.0f};

        for (int k = 0; k < a_col_b_row; k++) {
			//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);

            double2 a = A[k * res_row + row]; //column-major!!!!!!
            double2 b = B[col * a_col_b_row + k];
            double real_part = a.x * b.x - a.y * b.y;
            double imag_part = a.x * b.y + a.y * b.x;
            sum.x += real_part;
            sum.y += imag_part;
			}
			
		C[col * res_row + row] = sum;
		}
	}
}

void L_triangular_complex_matrix_mult(const double2* A, const double2* B, double2* C, const int res_row, const int a_col_b_row, const int res_col) {

	for(int row = 0; row<res_row; row++){
		for(int col = 0; col<res_col; col++){
			double2 sum = {.x = 0.0f, .y = 0.0f};
			if(row>=col){
				for (int k = 0; k < a_col_b_row; k++) {
					//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);

					double2 a = A[k * res_row + row]; //column-major!!!!!!
					double2 b = B[col * a_col_b_row + k];
					double real_part = a.x * b.x - a.y * b.y;
					double imag_part = a.x * b.y + a.y * b.x;
					sum.x += real_part;
					sum.y += imag_part;
					}
			}
			
		C[col * res_row + row] = sum;
		}
	}
}

/*double2 csqrt(double2 z){
	double complex z = z.x + z.y*I;
	double r = cabsf(z);
    double theta = atan2(z.y,z.x);
    double2 sqrt_z = {.x = sqrtf(r) * cosf(theta / 2.0f), .y = sqrtf(r) * sinf(theta / 2.0f)}//;make_cudoubleComplex(sqrtf(r) * cosf(theta / 2.0f),
                                                //sqrtf(r) * sinf(theta / 2.0f));
	return sqrt_z;
}*/

void bChol(double2* A,int i,int N){
	double complex z = A[i*N+i].x + A[i*N+i].y*I;
	z = csqrtf(z);
	double2 a = {.x = creal(z),.y = cimag(z)};
	A[i*N+i] = a;
}

void bChol2(double2* A,int i,int N){
	//int row = blockIdx.x + 1;
	//int diagonal = i*N+i;
	/*if(row == 1){
		//printf("sqtr %d \n",i*N+i); this is a 
		A[i*N+i] = cuCsqrt(A[i*N+i]);
	}*/
	//__syncthreads();
	//printf("\n %d %d",i*N+i, (i*N+i)+row);
	int range = N-i-1;//N-(i+1);
	for(int row = 0; row<range; row++){
		double2 z = {.x = (A[(i*N+i)+row+1].x * A[i*N+i].x + A[(i*N+i)+row+1].y * A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y), .y = (A[(i*N+i)+row+1].y * A[i*N+i].x  - A[(i*N+i)+row+1].x*A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y)};//cuCdivf(A[(i*N+i)+row], A[i*N+i]);
		A[(i*N+i)+row+1] = z;//(A[(i*N+i)+row].x*A[i*N+i].x + A[(i*N+i)+row].y * A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y);//cuCdivf(A[(i*N+i)+row], A[i*N+i]);
	}
}

void bChol3(double2* A, int i, int N){
	int j = i+1;
	int range = N-(i+1);
	//int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	//int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	//int vector = threadIdx.y + blockDim.y * blockIdx.y;
	for(int row = 0; row < range; row++){
		for(int col = 0; col < range; col++){
			if(row >= col){
				//printf("vec1 %d vec2 %d (%d,%d) index %d \n",(N*i+i+1)+row,(N*i+i+1)+col,row,col,(col+j)*N+j+row);
				//printf("%d = %d-%d*%d \n",(col+j)*N+j+row,(col+j)*N+j+row,(N*i+i+1)+row,(N*i+i+1)+col);
				double2 tmp = A[(N*i+i+1)+col];
				tmp.y = -tmp.y;
				double2 mult = {.x=(A[(N*i+i+1)+row].x*tmp.x - A[(N*i+i+1)+row].y*tmp.y), .y=(A[(N*i+i+1)+row].x*tmp.y + A[(N*i+i+1)+row].y*tmp.x)};
				double2 sub = {.x=(A[(col+j)*N+j+row].x-mult.x),.y=A[(col+j)*N+j+row].y-mult.y};
				A[(col+j)*N+j+row] = sub;//(double2){.x=A[(col+j)*N+j+row].x - A[(N*i+i+1)+row].x * A[(col+j)*N+j+row].y, .y=A[(col+j)*N+j+row].y - A[(N*i+i+1)+row].x * A[(col+j)*N+j+row].x - A[(N*i+i+1)+row].y * A[(col+j)*N+j+row].y};
				//A[(col+j)*N+j+row] = cuCsubf(A[(col+j)*N+j+row],cuCmulf(A[(N*i+i+1)+row],tmp));
			}
			
		}
		
	}
	
}

int main() {

	//read the Y.csv
	int num_rows = 128;
	int num_cols = 1;
	double2 *Y;
	char file[] = "128x8/Y";
	Y = read_matrix_from_csv(file, num_rows, num_cols);

	//read H.csv
	num_rows = 128;
	num_cols = 8;
	double2 *H;
	strcpy(file, "128x8/H");
	H = read_matrix_from_csv(file, num_rows, num_cols);

	//Size of N=antennas (nr of rows), K=Users (nr of columns)
	int N = 128;
	int K = 8;
	
	printf("---------------H---------------------------\n");
	for(int i=0; i<N; i++){//rows
		for(int j = 0; j<K; j++){//cols
			printf("%.9f+%.9fi	",H[j*N + i].x,H[j*N + i].y);//N*K
		}
		printf("\n");
	}
	printf("-------------------------------------------\n");

/*
	double2 inv[K*K];
	double2 HHY[N];
	*/
	double2 *HH, *mHH, *dInv, *dInvH,*dInvM,*dY,*HHY,*dx;
	HH = (double2 *) malloc(K * N * sizeof(double2));
	mHH = (double2 *) malloc(K * K * sizeof(double2));
	HHY = (double2 *) malloc(N * 1 * sizeof(double2));
	//mHH = (double2 *) malloc(K * K * sizeof(double2));
	
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
		
		printf("chol part 1------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%f+%fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*1
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");		
			
		bChol2(mHH,i,K);
		
				printf("chol part 2------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%f+%fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*1
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");		
			
		//bChol<<<1,1>>>(mHH,i,N);
		//cudaDeviceSynchronize();
		//Part2 of cholesky (column compleeted)
		//the amount of threads is getting smaler each itteration
		//it is the number of elements in the vector under the diagonal element
		//bChol2<<<K-(i+1),1>>>(dmHH,i,N);
		//cudaDeviceSynchronize();
		//Part3 of cholesky and start cInv part1
	//	cInv1<<<i+1,1>>>(dmHH,dInv,i,N);
	//	blockDims.x = N-(i+1);
	//	blockDims.y = N-(i+1);
		//bChol3<<<blockDims,1>>>(dmHH,i,N);
		bChol3(mHH,i,K);
		
				printf("chol part 3------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%f+%fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*1
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");		
			
		//cudaDeviceSynchronize();
		//Part2 of inv
		//blockDims.x = N-(i+1);
		//blockDims.y = N;
	//	cInv2<<<blockDims,1>>>(dmHH,dInv,i,N);
		//printf("\n");*/
	}
	
	
	printf("cholesky------------------------------------------\n");
	for(int i=0; i<K; i++){//rows
		for(int j = 0; j<K; j++){//cols
			printf("%f+%fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*K
		}
		printf(" ; \n");
	}
	printf("------------------------------------------\n");
}