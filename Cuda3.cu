#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef INFINITY
/* INFINITY is supported */
#endif

float **A, **D, *d2;
double elapsed_time;

void makeAdjacency(int n, float p, int w);

__global__ void calc(float *d_D, int n, int k){ //kernel (4  cells for every thread)
	__shared__ float s_d[4*3*256]; //Shared table within a block
	int i = blockIdx.x * blockDim.x + threadIdx.x; //Calculation of i and j
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int b_index = 4 * 3 * (threadIdx.x + blockDim.x*threadIdx.y); //Calculation of initial index of thread in the shared table within the block
	int istep = blockDim.x*gridDim.x, jstep = blockDim.y*gridDim.y;
	int l, m , v=0;
	for (l = 0; l<2; l++){
		for (m = 0; m<2; m++){ //Pass values from device table to shared block table for every one of the 4 cells
			s_d[b_index + 3 * v] = d_D[(i+l*istep)+(j+m*jstep)*n]; 
			s_d[b_index + (3 * v + 1)] = d_D[(i + l*istep) + k*n];
			s_d[b_index + (3 * v + 2)] = d_D[k + (j + m*jstep)*n];
			v++;
		}
	}
	for (v = 0; v<4; v++){ //Calculate the new cell values (4 for every thread)
		if (s_d[b_index + 3 * v] > s_d[b_index + (3 * v + 1)] + s_d[b_index + (3 * v + 2)]) s_d[b_index + 3 * v] = s_d[b_index + (3 * v + 1)] + s_d[b_index + (3 * v + 2)];
	}
	v = 0;
	for (l = 0; l<2; l++){ //Pass the new values to the device table
		for (m = 0; m<2; m++){
			d_D[(i+l*istep)+(j+m*jstep)*n] = s_d[b_index + 3 * v];
			v++;
		}
	}
}

int main(int argc, char **argv){
	int N, w;
	float p;

	N = atoi(argv[1]); //Read values from console
	p = atof(argv[2]);
	w = atoi(argv[3]);
	int n = pow(2, N);
	makeAdjacency(n, p, w); //Initialize distance values
	clock_t start = clock();
	int i, j, k;
	D = (float **)malloc(n*sizeof(float *)); //Allocation and initialization of D
	for (i = 0; i<n; i++) D[i] = (float *)malloc(n*sizeof(float));
	for (i = 0; i<n; i++){
		for (j = 0; j<n; j++){ 
			D[i][j] = INFINITY;
			if ((!isinf(A[i][j])) && A[i][j] != 0) D[i][j] = A[i][j];
			if (A[i][j] == 0) D[i][j] = 0;
		}
	}
	float *d_D;
	d2 = (float *)malloc(n*n*sizeof(float)); //Pass the values in the 1-d table
	int index = 0;
	for (j = 0; j<n; j++){
		for (i = 0; i<n; i++){
			d2[index++] = D[i][j];
		}
	}
        int gridx = pow(2, N - 4)/2, gridy = pow(2, N - 4)/2; //size of grid (n/4)*4 cells/thread=n cells
	int blockx = pow(2, 4), blocky = pow(2, 4);
	dim3 dimGrid(gridx, gridy);
	dim3 dimBlock(blockx, blocky);
	int size = n*n*sizeof(float);
	cudaMalloc((void**)&d_D, size); //Allocation of the device table
	cudaMemcpy(d_D, d2, size, cudaMemcpyHostToDevice); //Pass values from host to device
	for (k = 0; k<n; k++){
		
	    calc<<<dimGrid,dimBlock>>>(d_D,n,k); //Run kernel k times
		
	}
	cudaMemcpy(d2, d_D, size, cudaMemcpyDeviceToHost); //Return to host
	cudaFree(d_D);
	index = 0;
	for (j = 0; j<n; j++){
		for (i = 0; i<n; i++){
			D[i][j]=d2[index++];  //Pass the values to final table D[i][j]
		}
	}
	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed wall time = %f sec\n", seconds); //Run wall time
	exit(0);
}

void makeAdjacency(int n, float p, int w){ //Function for initialization of distance table A
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


