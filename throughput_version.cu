#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>

/*
	128x8		
	1024x64
    4096x128

	Blocks 	2 4 8 16 32 64 128
	Grid	1  2  4  8
	everything square matrix
	Vectors get the same number as blocks
*/

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
	  char line[8192];
	  int row = 0, col = 0;
	  while (fgets(line, 8192, file) && row < num_rows) {
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
	exrta calculates how many elements of a matrix each thread needs to calculate of there are to few threads
	elements is the number of elements there is in a row/column in the matrix.
	nrOfThreads are the number of threads avaleble for use
*/
__device__ int extra(int elements, int nrOfThreads){
	/*int extraLoops = 1;
	while(nrOfThreads*extraLoops < elements){
		extraLoops++;
	}
	return extraLoops;*/
	int tmp = ceil((float)elements/(float)nrOfThreads);
	return tmp;
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
	It takes the unfinished rows and subtract them with
	the multiplication of the ith column element in the row and the i row. 
	A is the matrix that is choleskylised
	i is the column that is calculated
	N is the nr of rows/columns of the A matrix (NxN)
	The A matrix is overwriten in this funktion
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
			//inte säker på = i första if satsen
			if(tmpRow <= rowElements && tmpCol <=N){
				if(tmpRow+i+1 >= tmpCol){
				//printf("P2 (%d,%d) %d\n",row+i+1,col,i);
				//printf(" %d-%d*%d",j*N+k,j*N+i,i*N+k);
				Ainv[tmpCol*N+tmpRow+i+1] = cuCsubf(Ainv[tmpCol*N+tmpRow+i+1],cuCmulf(Ainv[tmpCol*N+i],A[i*N+tmpRow+i+1]));
				//Ainv[col*N+row] = Ainv[col*N+row]-Ainv[col*N+i]*A[i*N+row];
				}
			}
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
	//printf("P1 (%d,%d) %d/%d \n",i,col,col*N+i,i*N+i);
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
   The A matrix is overwriten in this funktion
*/
__global__ void bChol3(float2* A, int i, int N){
	
	//N-(i+1) is the number of elements run in both x and y
	int elements = N-(i+1);
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(elements, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(elements, colthread);
	
	int j = i+1;
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	//int vector = threadIdx.y + blockDim.y * blockIdx.y;
	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
				int tmpRow = row+rowthread*v;
				int tmpCol = col+colthread*w;
			if(tmpRow >= tmpCol && tmpRow<=elements && tmpCol<=elements){
				//printf("vec1 %d vec2 %d (%d,%d) index %d \n",(N*i+i+1)+row,(N*i+i+1)+col,row,col,(col+j)*N+j+row);
				//printf("%d = %d-%d*%d \n ",(col+j)*N+j+row,(col+j)*N+j+row,(N*i+i+1)+row,(N*i+i+1)+col);
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
   The A matrix is overwriten in this funktion
*/
__global__ void bChol2(float2* A,int i,int N){
	
	int rowthread = blockDim.x * gridDim.x;
	//N-(i+1) is the number of elements calculated in this part
	int elements = N-(i+1);
	int extraRows = extra(elements, rowthread);
	
	int row = blockIdx.x + 1;
	
	for(int v=0;v < extraRows; v++){
		int tmpRow = row+rowthread*v;
		if(tmpRow <= elements){
			//printf("(%d*%d) %d %d,  \n",tmpRow,i*N+i, extraRows,elements);
			A[(i*N+i)+tmpRow] = cuCdivf(A[(i*N+i)+tmpRow], A[i*N+i]);
		}
	}
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
	
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(K, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(N, colthread);
	
	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;	//ex 0,0 or 1,3
	
	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
			int tmpRow = row+rowthread*v;
			int tmpCol = col+colthread*w;
			if(tmpRow < K && tmpCol <N){
				
				int idx_in = tmpCol * K + tmpRow;
				int idx_out = tmpRow * N + tmpCol;
				//printf("(%d,%d)  in: %d, out: %d\n",tmpRow,tmpCol,idx_in,idx_out);
				//conjugate here - in a float2: .x is the real part, .y is imaginary part
				output_hh[idx_out].x = input_h[idx_in].x; //conjugate
				output_hh[idx_out].y = -input_h[idx_in].y; //conjugate, it is negative for the imaginary part
			}
		}
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
	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(res_row, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(res_col, colthread);

	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;

	for(int v=0;v < extraRows; v++){
			for(int w=0; w<extraCols;w++){
				int tmpRow = row+rowthread*v;
				int tmpCol = col+colthread*w;
				
				if (tmpRow < res_row && tmpCol < res_col) {		
					float2 sum = make_float2(0.0f, 0.0f);
				
				for (int k = 0; k < a_col_b_row; k++) {
					//printf("(%d,%d) a: %d   b: %d\n",row,col,row * a_col_b_row + k, k * res_col + col);
					float2 a = A[k * res_row + tmpRow]; //column-major!!!!!!
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

	int rowthread = blockDim.x * gridDim.x;
	int extraRows = extra(res_row, rowthread);
	int colthread = blockDim.y * gridDim.y;
	int extraCols = extra(res_col, colthread);

	int row = threadIdx.x + blockDim.x * blockIdx.x; //find what col and row this thread is responsible for
	int col = threadIdx.y + blockDim.y * blockIdx.y;


	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCols;w++){
			int tmpRow = row+rowthread*v;
			int tmpCol = col+colthread*w;
			if (tmpRow < res_row && tmpCol < res_col && tmpRow >= tmpCol) {		
				float2 sum = make_float2(0.0f, 0.0f);
				for (int k = 0; k < a_col_b_row; k++) {
					//printf("(%d,%d) a: %d   b: %d\n",tmpRow,tmpCol,tmpRow * a_col_b_row + k, k * res_col + tmpCol);
					float2 a = A[k * res_row + tmpRow]; //column-major!!!!!!
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

int main() {
	//read the Y.csv
	//128x8
	int K,N,blockSize,gridSize;
	int nrOfRunns = 1;
	int nrOfFrames;
	
	printf("Enter N K blockSize gridSize nrOfFrames\n");
    scanf("%d %d %d %d %d",&K,&N,&blockSize,&gridSize,&nrOfFrames);

	
	printf("Info: %dx%d, blockSize=%d, gridSize=%d, nrOfRunns=%d nrOfFrames=%d\n",K,N,blockSize,gridSize,nrOfRunns,nrOfFrames);
	
	float times[nrOfRunns];
	
	char file1[32] = "";
	sprintf(file1, "%dx%d/Y", K,N);
	float2 *hY;
	hY = read_matrix_from_csv(file1, K, 1);
	//read H.csv
	float2 *H;
	sprintf(file1, "%dx%d/H", K,N);
	H = read_matrix_from_csv(file1, K, N);
	
	//Time stuff
	/*cudaEvent_t start[nrOfFrames], stop[nrOfFrames];     		// using cuda events to measure time
	float elapsed_time_ms[nrOfFrames];       		// which is applicable for asynchronous code also
	
	for(int frame = 0; frame<nrOfFrames; frame++){
		cudaEventCreate(&start[frame]);     		// instrument code to measure start time
		cudaEventCreate(&stop[frame]);
		
	}*/
	
	cudaEvent_t start, stop;     		// using cuda events to measure time
	float elapsed_time_ms;       		// which is applicable for asynchronous code also
	
	cudaEventCreate(&start);     		// instrument code to measure start time
	cudaEventCreate(&stop);
		
	//The h stands for host
	float2 *hHHY[nrOfFrames];
	
	for(int frame = 0; frame<nrOfFrames; frame++){
		hHHY[frame] = (float2 *) malloc(N * sizeof(float2));;
	}
	
	cudaEventRecord(start, 0);//start time
	for(int o=0; o<nrOfRunns;o++){
		//The d stands for device
		cuFloatComplex *dH[nrOfFrames], *dHH[nrOfFrames], *dmHH[nrOfFrames], *dInv[nrOfFrames], *dInvH[nrOfFrames],*dInvM[nrOfFrames],*dY[nrOfFrames],*dHHY[nrOfFrames],*dx[nrOfFrames];
		for(int frame = 0; frame<nrOfFrames; frame++){
			cudaMalloc((void **)&dH[frame], K*N*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dHH[frame],  N*K*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dmHH[frame], N*N*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dInv[frame], N*N*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dInvH[frame], N*N*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dInvM[frame], N*N*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dY[frame], K*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dHHY[frame], K*sizeof(cuFloatComplex));
			cudaMalloc((void **)&dx[frame], K*sizeof(cuFloatComplex));
		}	
			
		for(int frame=0; frame<nrOfFrames;frame++){
			//Copy input data to array on GPU.
			cudaMemcpy(dH[frame], H, K*N*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
			cudaMemcpy(dY[frame], hY, K*sizeof(cuFloatComplex), cudaMemcpyHostToDevice);
			
		}
	
		//Run the transpose on gpu
		//Number of threads are K*N with k-rows and N-columns
		dim3 blockDims(blockSize,blockSize);
		dim3 GridDims(gridSize,gridSize);
		
		for(int frame = 0; frame<nrOfFrames; frame++){
			hermitian_transpose<<<blockDims,GridDims>>>(dH[frame], dHH[frame],K,N);
		}

		for(int frame = 0; frame<nrOfFrames; frame++){
		//Number of threads are N*N
			Ltriangle_complex_matrix_mult<<<blockDims,GridDims>>>(dHH[frame], dH[frame], dmHH[frame],N,K,N);	
		}
		cudaDeviceSynchronize();
		
		//cudaMemcpy(hmHH, dmHH, N*N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
		
		for(int i = 0; i < N; i++){
			//Del1 of cholesky. (Diagonal element) one thread
			for(int frame = 0; frame<nrOfFrames; frame++){
				bChol<<<1,1>>>(dmHH[frame],i,N);
			}
			cudaDeviceSynchronize();
			//Part2 of cholesky (column compleeted)
			//the amount of threads is getting smaler each itteration
			//it is the number of elements in the vector under the diagonal element
			for(int frame = 0; frame<nrOfFrames; frame++){
				bChol2<<<blockSize,1>>>(dmHH[frame],i,N);
			}
			cudaDeviceSynchronize();
			//Part3 of cholesky and start cInv part1
			for(int frame = 0; frame<nrOfFrames; frame++){
				cInv1<<<blockSize,1>>>(dmHH[frame],dInv[frame],i,N);
				bChol3<<<blockDims,GridDims>>>(dmHH[frame],i,N);
			}
			cudaDeviceSynchronize();
			//Part2 of inv
			for(int frame = 0; frame<nrOfFrames; frame++){
				cInv2<<<blockDims,GridDims>>>(dmHH[frame],dInv[frame],i,N);
			}
			//printf("\n");
		}	
		
		for(int frame = 0; frame<nrOfFrames; frame++){
			//This part takes the inv of L multiplied with itsef to become A^-1
			hermitian_transpose<<<blockDims,GridDims>>>(dInv[frame], dInvH[frame],N,N);
		}
		cudaDeviceSynchronize();
		for(int frame = 0; frame<nrOfFrames; frame++){
			//complex_matrix_mult<<<blockDims,GridDims>>>(dInv, dInvH, dInvM,K,K,K); Right way but not for Ali
			complex_matrix_mult<<<blockDims,GridDims>>>(dInvH[frame], dInv[frame], dInvM[frame],N,N,N);
		
			//dHH = 8x128 dy = 128x1 dHHY = 8x1
			complex_matrix_mult<<<blockDims,GridDims>>>(dHH[frame], dY[frame], dHHY[frame],N,K,1);
		}
		cudaDeviceSynchronize();
		for(int frame = 0; frame<nrOfFrames; frame++){
			//dHH = 8x8 dHHY = 8x1
			complex_matrix_mult<<<blockDims,GridDims>>>(dInvM[frame], dHHY[frame], dx[frame],N,N,1);		
		}
		for(int frame = 0; frame<nrOfFrames; frame++){
			cudaMemcpy(hHHY[frame], dx[frame], N*sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
		}
		
		cudaEventRecord(stop, 0);     	// instrument code to measue end time
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed_time_ms, start, stop);
		//printf("Time to calculate results on GPU: %f ms.\n", elapsed_time_ms);
		times[o] = elapsed_time_ms;
		
		//PRINT RESULT_________________________________________________________
		/*
		float2 temp[N];
		for (int frame = 0; frame<nrOfFrames; frame++){
			//printf("%d \n", i);
			memcpy(temp,hHHY[frame],N*sizeof(cuFloatComplex));
			//temp = *hHHY[frame]; 
			for(int i = 0; i<N; ++i) {
				printf("%f+%fi \n", temp[i].x,temp[i].y);//(*hHHY[i*N + frame]).x,(*hHHY[i*N+frame]).y);
			}
		}*/
		
		printf("Time to calculate results on GPU: %f ms or %f each.\n", elapsed_time_ms,elapsed_time_ms/nrOfFrames);
		
		// Free up the arrays on the GPU.
		cudaFree(dH);
		cudaFree(dHH);
		cudaFree(dmHH);
		cudaFree(dInv);
		cudaFree(dInvH);
		cudaFree(dInvM);
		cudaFree(dY);
		cudaFree(dHHY);
	}
	//Free from CPU
	free(hY);
	free(H);
	/*
	float mean = 0;
	qsort(times, nrOfRunns, sizeof(int), cmpfunc);
	for (int i = 0; i<nrOfRunns; ++i) {
		mean += times[i];
		//printf("%f \n", times[i]);
	}
	mean = mean/nrOfRunns;
	printf("%dx%d(%d,%d) Mean: %f Median: %f Min: %f Max: %f \n", K,N,blockSize,gridSize,mean,times[nrOfRunns/2],times[0],times[nrOfRunns-1]);
	printf("%dx%d(%d,%d) %f \n",K,N,blockSize,gridSize,times[nrOfRunns/2]);*/
    return 0;
}
