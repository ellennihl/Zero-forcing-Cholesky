#include <stdio.h>
#include <math.h>

void bChol(double A[], int i, int N);
void bCholPart2(double A[], int i, int N);
void bCholPart3(double A[], int i, int N);

//Step 1 sqrt(A_ii);
//A[] is the matrix that is decomposed
//i is the column of the matrix that is being decomopes
//N is the size of the N*N matrix
void bChol(double A[], int i, int N){
		A[i*(N+1)] = sqrt(A[i*(N+1)]);
		bCholPart2(A,i,N);
		bCholPart3(A,i,N);
}

//This part can be parallelised
//Step2 2 A_ij/A_ii
void bCholPart2(double A[], int i, int N){
		for(int j = 1; j < N-i; j++){
			A[i*(N+1)+j] = A[i*(N+1)+j]/A[i*(N+1)];
		}
}

//This part can be parallelised
//Step3 U-c*c^T
void bCholPart3(double A[], int i, int N){
		for(int k = i+1; k< N; k++){
			for(int j = k; j< N; j++){
				A[k*N+j] = A[k*N+j] - A[i*N+k]*A[i*N+j];
			}
		}
}

int main(int argc, char *argv[])  {
	// Perform Block Cholesky on A = L*L^T
    // where A and L are NxN matrices
    int N = 3;
    int size = N*N;
	double A[size]
	/*double L[size]; Just nu blir A till L
	for (int i = 0; i < size; i++){
		L[i] = 0;
	}*/
	
	//Initialize
	/*
		2	2	3	
		2	10	5
		3	5	20
	*/
	
	A[0] = 2;
	A[1] = 2;
	A[2] = 3;
	A[3] = 2;
	A[4] = 10;
	A[5] = 5;
	A[6] = 3;
	A[7] = 5;
	A[8] = 20;
	
	//Initialize ÄNDRA N OM DU SKA BYTA
	/*
		2	2	3	4
		2	10	5	6
		3	5	20	7
		4	6	7	30
	*/
	/*
	A[0] = 2;
	A[1] = 2;
	A[2] = 3;
	A[3] = 4;
	A[4] = 2;
	A[5] = 10;
	A[6] = 5;
	A[7] = 6;
	A[8] = 3;
	A[9] = 5;
	A[10] = 20;
	A[11] = 7;
	A[12] = 4;
	A[13] = 6;
	A[14] = 7;
	A[15] = 30;
	*/
	
	//Do the cholesky column by column
	for(int i = 0; i < N; i++){
		bChol(A,i,N);
	}
	
	//print the matrix
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++){
			//printf("%d ",j*N+i);
			printf("%f ",A[j*N+i]);
		}
		printf("\n");
	}
}