#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef INFINITY
/* INFINITY is supported */
#endif

float **A, **D;      //Table A distance, D minimum distance
void makeAdjacency(int n, float p, int w);  //Set initial values to node distances Α

int main(int argc, char **argv){
	int N, w;     
	float p;
	N = atoi(argv[1]);  //Read the console inputs
	p = atof(argv[2]);
	w = atoi(argv[3]);
	int n = pow(2, N);
	makeAdjacency(n, p, w);  //Initialize table A
	clock_t start = clock();  //First time measurement
    int i, j, k;
	D = (float **)malloc(n*sizeof(float *));  //Allocation for table D
	for (i = 0; i<n; i++) D[i] = (float *)malloc(n*sizeof(float));
	for (i = 0; i<n; i++){   //Initial values for D
		for (j = 0; j<n; j++){
			D[i][j] = INFINITY;			
			if ((!isinf(A[i][j])) && A[i][j] != 0) 	D[i][j] = A[i][j];
			if (A[i][j] == 0) D[i][j] = 0;
		}
	}
	for (k = 0; k<n; k++){   //Calculate minimun distances based on the algorithm
		for (i = 0; i<n; i++){
			for (j = 0; j<n; j++){
				if (D[i][j] > D[i][k] + D[k][j]) D[i][j] = D[i][k] + D[k][j];
			}
		}
	}
	clock_t end = clock();   //Final time calculation and convert it into seconds
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Elapsed wall time = %f sec\n", seconds);
}

void makeAdjacency(int n, float p, int w){   //Set initial values to node distances
	int i, j;
	A = (float **)malloc(n*sizeof(float *));   //Allocation of table A
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
