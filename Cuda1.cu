#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef INFINITY
/* INFINITY is supported */
#endif

float **A, **D, *d2; //Table A distance, D minimum distance,d2 tempTable 1-d

void makeAdjacency(int n, float p, int w);  //kernel
__global__ void calc(float *d_D, int n, int k){
	int i = blockIdx.x * blockDim.x + threadIdx.x;   //We find i & j in the Grid of threads
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (d_D[i + j*n] > d_D[i + k*n] + d_D[k + j*n]) d_D[i + j*n] = d_D[i + k*n] + d_D[k + j*n];  //Every thread calculates its proper value
}

int main(int argc, char **argv){
	int N, w;
	float p;

	N = atoi(argv[1]);   //Read the console inputs
	p = atof(argv[2]);
	w = atoi(argv[3]);
	int n = pow(2, N);
	makeAdjacency(n, p, w);  //Initialize table A
	clock_t start = clock();  //First time measurement
	int i, j, k;
	D = (float **)malloc(n*sizeof(float *));   //Allocation for table D
	for (i = 0; i<n; i++) D[i] = (float *)malloc(n*sizeof(float));   
	for (i = 0; i<n; i++){   //Initial values for D
		for (j = 0; j<n; j++){
			D[i][j] = INFINITY;
			if ((!isinf(A[i][j])) && A[i][j] != 0) {
				D[i][j] = A[i][j];
			}
			if (A[i][j] == 0) D[i][j] = 0;
		}
	}
	d2 = (float *)malloc(n*n*sizeof(float));  //Pass the values into the subtable d2
	int index = 0;
	for (j = 0; j<n; j++){
		for (i = 0; i<n; i++){
			d2[index++] = D[i][j];
		}
	}
	int gridx = pow(2, N - 4), gridy = pow(2, N - 4);  //Dimensions of grid
	int blockx = pow(2, 4), blocky = pow(2, 4);
	dim3 dimGrid(gridx, gridy);
	dim3 dimBlock(blockx, blocky);
	int size = n*n*sizeof(float);
	float *d_D;
	cudaMalloc((void**)&d_D, size);  //Allocation of device Table
	cudaMemcpy(d_D, d2, size, cudaMemcpyHostToDevice); //Memory transfer from host to device
	for (k = 0; k<n; k++){		
		calc << <dimGrid, dimBlock >> >(d_D, n, k);  //Run kernel for each k
	}
	cudaMemcpy(d2, d_D, size, cudaMemcpyDeviceToHost);  //Pass values from device to host
	cudaFree(d_D);
	index = 0;
	for (j = 0; j<n; j++){
		for (i = 0; i<n; i++){
			D[i][j] = d2[index++];  //Pass the values to the 2-d Table of min distance D[i][j]
		}
	}
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed wall time = %f sec\n", seconds);  //Elapsed time 
	exit(0);
}

void makeAdjacency(int n, float p, int w){  //Set initial values to node distances
	int i, j;
	A = (float **)malloc(n*sizeof(float *));
	for (i = 0; i<n; i++) A[i] = (float *)malloc(n*sizeof(float));
	srand(time(NULL));
	for (i = 0; i<n; i++){
		for (j = 0; j<n; j++){
			if (((float)rand() / (RAND_MAX)) > p) {
				A[i][j] = INFINITY;
			}
			else A[i][j] = ((float)rand() / (RAND_MAX)) * w;
		}
		A[i][i] = 0;
	}
}


