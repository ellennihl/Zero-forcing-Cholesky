#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuComplex.h>

/*
 Thread use
	This is a recap of how many elements that are calculated in each step
	Size of matrix NxK = 128x8
	
	Overal the biggest matrix is when taking the hermetian of the NxK matrix
	
		1 Hermetian transpose
			Full matrix
			threads = NxK = 128x8
		2 Matrix multiplication
			Calculatring H*H^H witch makes a KxK
			threads = KxK = 8x8
		3 Cholesky
			The biggest matrix here is K-1xK-1
		3.1 CholPart1
			Sqrt of diagonal element
			threads = 1x1
		3.2 CholPart2
			Deviding the rest of the column under the diagonal element
			The biggest one one is K-1x1 and smalest 1x1
			threads = K-1x1 = 7x1
		3.2 CholPart3
			The rest of the matrix after part1 and 2
			Starts by K-1xK-1 and down to 1x1
			threads = K-1xK-1 = 7x7
		4 Inversion
			The biggers one here is the K-1xK
		4.1 InvPart1
			The diagonal row is calculated
			Starts at 1x1 and calculates upp to Kx1
			threads = Kx1 = 8x1
		4.2 InvPart2
			Calculates the rest of the rows
			Starts at K-1xK and calculates down to 1xK
			threads = K-1xK = 7x8
		5 Hermetian transpose
			Calculates the hemetian of L^-1
			threads = KxK = 8x8
		6 Matrix multiplication
			Getting A^-1 = LH^-1*L^-1
			threads = KxK = 8x8
		7 Matrix multiplication
			A^-1*H^H*Y
			threads = Kx1 = 8x1
*/	

__global__ void treadtester(int i,int j,int* is){
	is[0]=is[0]+1;
	bool rowSmall = false, colSmall = false;
	int extraRows = 1;
	int extraCol = 1;
	
	//check if we got more or less threads than elements
	int rowthread = blockDim.x * gridDim.x;
	while(rowthread*extraRows < i){
		rowSmall = true;
		extraRows++;
	}

	int colthread = blockDim.y * gridDim.y;
	while(colthread*extraCol < j){
		colSmall = true;
		extraCol++;
	}
	
	int row = threadIdx.x + blockDim.x * blockIdx.x; 
	int col = threadIdx.y + blockDim.y * blockIdx.y;
	for(int v=0;v < extraRows; v++){
		for(int w=0; w<extraCol;w++){
			printf("(%d,%d) %d %d id%d\n",row*rowthread+v,col*colthread+w,rowSmall,colSmall,(row+v*rowthread)+(col+w*colthread)*rowthread*extraRows);
		}
	}
		
}

int main(){
	int *myInt;
	cudaMalloc((void **)&myInt, sizeof(int));
	printf("start \n");
	dim3 blockDims(1,2);
	dim3 GridDims(2,1);
	treadtester<<<blockDims,GridDims>>>(2,8,myInt);
	
	cudaFree(myInt);
	return 0;
}