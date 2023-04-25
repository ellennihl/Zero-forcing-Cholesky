#include <stdio.h>
#include <math.h>

void bChol(double A[], int i,int index, int N);
void bCholPart2(double A[], int i,int index, int N);
void bCholPart3(double A[], int i,int index, int N);

//Step 1 sqrt(A_ii);
//A[] is the matrix that is decomposed
//i is the column of the matrix that is being decomopes
//index is the index of diagonal element of the column i
//N is the size of the N*N matrix
void bChol(double A[], int i, int index, int N){
		A[index] = sqrt(A[index]);
		bCholPart2(A,i,index,N);
		bCholPart3(A,i,index,N);
		//printf("\n");
}

//This part can be parallelised
void bCholPart2(double A[], int i,int index, int N){
		//Step 2 A_ij/A_ii
		for(int j = 1; j < N-i; j++){
			//printf("%d ",index+j);
			A[index+j] = A[index+j]/A[index];
		}
}

//This part can be parallelised
void bCholPart3(double A[], int i,int index, int N){
		//temp är index av nästa kolumns diagonala element
		//k är antal columner efter i 
		//j går igenom 
		//Step 3 U-c*c^T
		int temp = index;
		for(int k = i; k< N; k++){
			temp += N-k;
			for(int j = 0; j< N-k-1; j++){
				//printf("%d %d,",index+k-i+j+1,index+k-i+1);
				A[temp+j] = A[temp+j] - A[index+j+k-i+1]*A[index+k-i+1];
			}
		}
}

int main(int argc, char *argv[])  {
		
	// Perform Cholesky on A = L*L^T
    // where A and L are NxN matrices
    int N = 4;
	int size = 0;
	for(int i = 1; i <= N; i++){
		size += i;
	}
	double A[size],L[size];

	//Initialize
	/*
		2	0	0	
		2	10	0
		3	5	20
	*/
	
	/*
		2	0	0  	0	
		2	10	0  	0
		3	5	20 	0
		4	6	7	30
	*/
	
	/*
		2	0	0  	0	0
		2	10	0  	0	0
		3	5	20 	0	0
		-	-	-	-	0
		-	-	-	-	-
	*/
	/*
	A[0] = 2;
	A[1] = 2;
	A[2] = 3;
	A[3] = 10;
	A[4] = 5;
	A[5] = 20;
	*/

	A[0] = 2;
	A[1] = 2;
	A[2] = 3;
	A[3] = 4;
	A[4] = 10;
	A[5] = 5;
	A[6] = 6;
	A[7] = 20;
	A[8] = 7;
	A[9] = 30;
	
	
	
	for (int i = 0; i < size; i++){
		L[i] = 0;
		//A[i] = 0;
	}
	
	//Do the cholesky column by column
	//int index = 0;
	for(int i = 0,index = 0; i < N; i++){
		bChol(A,i,index,N);
		index += N-i;
	}
	

	//print the matrix (har inte en fin print funktion än)
	/*
		0	-	-
		1	3	-
		2	4	5
		
		prints like 0 1 2 3 4 5
	*/	
	for (int i = 0; i < size; i++) {
		printf("%f ",A[i]);
	}
}