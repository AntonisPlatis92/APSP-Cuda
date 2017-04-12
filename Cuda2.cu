#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef INFINITY
/* INFINITY is supported */
#endif

float **A, **D, *d2;  //Table A distance, D minimum distance,d2 tempTable 1-d

void makeAdjacency(int n, float p, int w);  

__global__ void calc(float *d_D, int n, int k){ //kernel
        __shared__ float s_d[3*256]; //shared in block table of floats (size 3*number threads/block)
	int i = blockIdx.x * blockDim.x + threadIdx.x;  //We find i & j in the Grid of threads
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int b_index = 3 * (threadIdx.x + blockDim.x*threadIdx.y); //Calculation of initial index in shared table s_d
	s_d[b_index] = d_D[i + j*n];  //Pass values from device table to shared
	s_d[b_index + 1] = d_D[i + k*n];
	s_d[b_index + 2] = d_D[k + j*n];
	if (s_d[b_index] > s_d[b_index + 1] + s_d[b_index + 2]) s_d[b_index] = s_d[b_index + 1] + s_d[b_index + 2]; //Calculation of new distance value
	d_D[i + j*n] = s_d[b_index]; //Pass the values back to the table s_d
}

int main(int argc, char **argv){
	int N, w;
	float p;

	N = atoi(argv[1]); //Read values from console
	p = atof(argv[2]);
	w = atoi(argv[3]);
	int n = pow(2, N);
	makeAdjacency(n, p, w);  //Initialize distance values
	clock_t start = clock();
	int i, j, k;
	D = (float **)malloc(n*sizeof(float *));  //Allocation and initialization of D
	for (i = 0; i<n; i++) D[i] = (float *)malloc(n*sizeof(float));
	for (i = 0; i<n; i++){
		for (j = 0; j<n; j++){
			D[i][j] = INFINITY;
			if ((!isinf(A[i][j])) && A[i][j] != 0) D[i][j] = A[i][j];
			if (A[i][j] == 0) D[i][j] = 0;
		}
	}
	float *d_D;
	d2 = (float *)malloc(n*n*sizeof(float));
	int index = 0;
	for (j = 0; j<n; j++){   //Pass the values in the 1-d table
		for (i = 0; i<n; i++){
			d2[index++] = D[i][j];
		}
	}
        int gridx = pow(2, N - 4), gridy = pow(2, N - 4); //Size of grid & block
	int blockx = pow(2, 4), blocky = pow(2, 4);
	dim3 dimGrid(gridx, gridy);
	dim3 dimBlock(blockx, blocky);
	int size = n*n*sizeof(float);
	cudaMalloc((void**)&d_D, size);  //Allocation of the device table
	cudaMemcpy(d_D, d2, size, cudaMemcpyHostToDevice); //Pass values from host to device
	for (k = 0; k<n; k++){
		
		calc << <dimGrid, dimBlock >> >(d_D, n, k); //Run the kernel k times
		
	}
	cudaMemcpy(d2, d_D, size, cudaMemcpyDeviceToHost); //Return values into the host table
	cudaFree(d_D);
	index = 0;
	for (j = 0; j<n; j++){
		for (i = 0; i<n; i++){
			D[i][j] = d2[index++]; //Pass the values in the final 2-d table
		}
	}
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed wall time = %f sec\n", seconds);  //Run wall time
	exit(0);
}

void makeAdjacency(int n, float p, int w){ //Initialization of distance of nodes
	int i, j;
	A = (float **)malloc(n*sizeof(float *));
	for (i = 0; i<n; i++) A[i] = (float *)malloc(n*sizeof(float));
	srand(time(NULL));
	for (i = 0; i<n; i++){
		for (j = 0; j<n; j++){
			if (((float)rand() / (RAND_MAX))>p) A[i][j] = INFINITY;
			else A[i][j] = ((float)rand() / (RAND_MAX)) * w;
		}
		A[i][i] = 0;
	}
}


