//serial execution
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <complex.h>
#include <time.h>
#include <cuComplex.h>
/*
typedef struct float2{
	float x; //real
	float y; //imaginary
} float2;*/

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
	  int size = num_rows*num_cols;
	  char line[size];//line[1024];
	  int row = 0, col = 0;
	  while (fgets(line, size, file) && row < num_rows) {//(fgets(line, 1024, file) && row < num_rows) {
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

cuFloatComplex cuCsqrt(cuFloatComplex z){
	float r = cuCabsf(z);
    float theta = atan2(z.y,z.x);
    cuFloatComplex sqrt_z = make_cuFloatComplex(sqrtf(r) * cosf(theta / 2.0f),
                                                sqrtf(r) * sinf(theta / 2.0f));
	return sqrt_z;
}

void bChol(float2* A,int i,int N){
	//float complex z = A[i*N+i].x + A[i*N+i].y*I;
	//z = csqrtf(z);
	//cuFloatComplex z = A[i*N+i];
	
	//float2 a = {.x = creal(z),.y = cimag(z)};
	//A[i*N+i] = a;
	A[i*N+i] = cuCsqrt(A[i*N+i]);
}


void bChol2(float2* A,int i,int N){
	int range = N-i-1;
	for(int row = 0; row<range; row++){
		float2 z = {.x = (A[(i*N+i)+row+1].x * A[i*N+i].x + A[(i*N+i)+row+1].y * A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y), .y = (A[(i*N+i)+row+1].y * A[i*N+i].x  - A[(i*N+i)+row+1].x*A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y)};//cuCdivf(A[(i*N+i)+row], A[i*N+i]);
		A[(i*N+i)+row+1] = z;
	}
}

void bChol3(float2* A, int i, int N){
	int j = i+1;
	int range = N-(i+1);
	for(int row = 0; row < range; row++){
		for(int col = 0; col < range; col++){
			if(row >= col){
				float2 tmp = A[(N*i+i+1)+col];
				tmp.y = -tmp.y;
				float2 mult = {.x=(A[(N*i+i+1)+row].x*tmp.x - A[(N*i+i+1)+row].y*tmp.y), .y=(A[(N*i+i+1)+row].x*tmp.y + A[(N*i+i+1)+row].y*tmp.x)};
				float2 sub = {.x=(A[(col+j)*N+j+row].x-mult.x),.y=A[(col+j)*N+j+row].y-mult.y};
				A[(col+j)*N+j+row] = sub;//(double2){.x=A[(col+j)*N+j+row].x - A[(N*i+i+1)+row].x * A[(col+j)*N+j+row].y, .y=A[(col+j)*N+j+row].y - A[(N*i+i+1)+row].x * A[(col+j)*N+j+row].x - A[(N*i+i+1)+row].y * A[(col+j)*N+j+row].y};
			}
			
		}
		
	}
	
}

void cInv1(float2* A,float2* Ainv, int i, int N){
	//int col = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	for(int col = 0; col<i+1; col++){
		if(col == i){
			Ainv[col*N+i].x = 1;
		}
		float2 div = {.x = (Ainv[col*N+i].x * A[i*N+i].x + Ainv[col*N+i].y * A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y), .y = (Ainv[col*N+i].y * A[i*N+i].x  - Ainv[col*N+i].x*A[i*N+i].y)/(A[i*N+i].x*A[i*N+i].x + A[i*N+i].y*A[i*N+i].y)};
		Ainv[col*N+i] = div;//cuCdivf(Ainv[col*N+i],A[i*N+i]);
	}
}

//blockDims.x = N-(i+1);
//blockDims.y = N;
void cInv2(float2* A,float2* Ainv, int i, int N){
	//int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	//int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	for(int row = 0; row < N-(i+1); row++){
		for(int col = 0; col < N; col++){
			if(row+i+1 >= col){
			//printf("P2 (%d,%d) %d\n",row+i+1,col,i);
			//printf(" %d-%d*%d",j*N+k,j*N+i,i*N+k);
			//Ainv[col*N+row+i+1] = cuCsubf(Ainv[col*N+row+i+1],cuCmulf(Ainv[col*N+i],A[i*N+row+i+1]));
			//Ainv[col*N+row] = Ainv[col*N+row]-Ainv[col*N+i]*A[i*N+row];
			
			float2 mult = {.x=(Ainv[col*N+i].x*A[i*N+row+i+1].x - Ainv[col*N+i].y*A[i*N+row+i+1].y), .y=(Ainv[col*N+i].x*A[i*N+row+i+1].y + Ainv[col*N+i].y*A[i*N+row+i+1].x)};
			float2 sub = {.x=(Ainv[col*N+row+i+1].x-mult.x),.y=Ainv[col*N+row+i+1].y-mult.y};
			Ainv[col*N+row+i+1] = sub;//(double2){.x=A[(col+j)*N+j+row].x - A[(N*i+i+1)+row].x * A[(col+j)*N+j+row].y, .y=A[(col+j)*N+j+row].y - A[(N*i+i+1)+row].x * A[(col+j)*N+j+row].x - A[(N*i+i+1)+row].y * A[(col+j)*N+j+row].y};

			}	
			
		}
		
	}
	
}

int main() {

	char str[20];
	char strY[20];
	char strH[20];
	
	int N, K, num_rows, num_cols, times;
	
	printf("Enter filename N K times\nExample: 4096x64 4096 64 10\n");
	scanf("%s %d %d %d",str,&N,&K,&times);
	
	clock_t start, end;
	double execution_time[times];

	for(int l = 0; l<times; l++){

		num_rows = N;
		num_cols = K;
		
		//read the Y.csv
		float2 *Y;
		
		strcpy(strY,str);
		strcpy(strH,str);
		strcat(strY, "/Y");
		strcat(strH, "/H");
		
		char file[20];// = str;//"1024x64/Y";//"128x8/Y";
		strcpy(file, strY);
		Y = read_matrix_from_csv(file, num_rows, 1);

		//read H.csv
		float2 *H;
		strcpy(file, strH);//"128x8/H");
		H = read_matrix_from_csv(file, num_rows, num_cols);

		start = clock();
		
		//Size of N=antennas (nr of rows), K=Users (nr of columns)
		//int N = 1024;//128;
		//int K = 64;//8;
		
		/*printf("---------------H---------------------------\n");
		for(int i=0; i<N; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%.9f+%.9fi	",H[j*N + i].x,H[j*N + i].y);//N*K
			}
			printf("\n");
		}
		printf("-------------------------------------------\n");
	*/
	/*
		float2 inv[K*K];
		float2 HHY[N];
		*/
		float2 *HH, *mHH, *invL, *invLH,*invM,*HHY,*x;
		HH = (float2 *) malloc(K * N * sizeof(float2));
		mHH = (float2 *) malloc(K * K * sizeof(float2));
		HHY = (float2 *) malloc(N * 1 * sizeof(float2));
		invL = (float2 *) malloc(K * K * sizeof(float2));
		invLH = (float2 *) malloc(K * K * sizeof(float2));
		invM = (float2 *) malloc(K * K * sizeof(float2));
		x = (float2 *) malloc(K * 1 * sizeof(float2));
		
		hermitian_transpose(H, HH, K, N);
		
	/*	printf("hermitian transpose------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<N; j++){//cols
				printf("%.9f+%.9fi	",HH[j*K + i].x,HH[j*K + i].y);//K*N
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");*/
		
		L_triangular_complex_matrix_mult(HH, H, mHH, K, N, K);
		/*
		printf("matmul------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%.9f+%.9fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*K
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");
		*/
		complex_matrix_mult(HH, Y, HHY, K, N, 1);
		
	/*	printf("matvecmul------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<1; j++){//cols
				printf("%f+%fi	",HHY[j*K + i].x,HHY[j*K + i].y);//K*1
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");*/
		
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
		//	cInv1<<<i+1,1>>>(dmHH,dInv,i,N);
			cInv1(mHH,invL,i,K);
		
		//	blockDims.x = N-(i+1);
		//	blockDims.y = N-(i+1);
			//bChol3<<<blockDims,1>>>(dmHH,i,N);
			bChol3(mHH,i,K);
			//cudaDeviceSynchronize();
			//Part2 of inv
			//blockDims.x = N-(i+1);
			//blockDims.y = N;
		//	cInv2<<<blockDims,1>>>(dmHH,dInv,i,N);
			cInv2(mHH,invL,i,K);
		//printf("\n");*/
		}
		
		hermitian_transpose(invL, invLH, K, K);
		complex_matrix_mult(invLH, invL, invM, K, K, K);
		complex_matrix_mult(invM, HHY, x, K, K, 1);
		
		end = clock();
		
		/*printf("cholesky------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%f+%fi	",mHH[j*K + i].x,mHH[j*K + i].y);//K*K
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");


		printf("inversion------------------------------------------\n");
		for(int i=0; i<K; i++){//rows
			for(int j = 0; j<K; j++){//cols
				printf("%f+%fi	",invL[j*K + i].x,invL[j*K + i].y);//K*K
			}
			printf(" ; \n");
		}
		printf("------------------------------------------\n");
	*/
		if(l==times-1){//print last iteration
			printf("x------------------------------------------\n");
			for(int i=0; i<K; i++){//rows
				for(int j = 0; j<1; j++){//cols
					printf("%f+%fi	",x[j*K + i].x,x[j*K + i].y);//K*1
				}
				printf(" ; \n");
			}
			printf("------------------------------------------\n");
		}

		execution_time[l] = ((double)(end - start))/CLOCKS_PER_SEC;//secs
		
		free(HH);
		free(mHH);
		free(HHY);
		free(invL);
		free(invLH);
		free(invM);
		free(x);
}	
	printf("for matrix %s \nduration: ",str);
	double sum = 0;
	for(int i=0; i<times; i++){
		printf("%f ", execution_time[i]);
		sum += execution_time[i];
	}
	printf("\nsum: %f \nclocks per sec: %ld\n", sum/times, CLOCKS_PER_SEC);
}