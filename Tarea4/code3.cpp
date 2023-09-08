%%sh
cat > vecadd.cu << EOF
#include <iostream>
#include <cmath>
#include <cuda.h>

using namespace std;

void matrixInit(float **h_x, int n) {
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			h_x[i][j] = i+j;
		}
	}
}

void vectInit(float *h_x, int n) {
	for(int i = 0; i < n; i++) {
		h_x[i] = i;
	}
}

void printMtx(float **h_x, int n) {
	cout << "[ ";
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			cout << h_x[i][j] << " ";
		}
		cout << "| \n";
	}
	cout << "] \n";
}

void printVect(float *h_x, int n) {
	cout << "[ ";
	for(int i = 0; i < n; i++) {
		cout << h_x[i] << " ";
	}
	cout << "] \n";
}

void toOneArray(float *h_nx, float **h_x, int n){
	int k = 0;
	
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			h_nx[k] = h_x[i][j];
			k++;
		}
	}
}

void toArrayFromOne(float *h_nx, float **h_x, int n){
	int k = 0;
	
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			h_x[i][j] = h_nx[k++];
		}
	}
}

__global__
void matAddKernel(float *d_A, float *d_B, float *d_C, int n) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i < n) {
		for(int k=0; k<n; k++){
			d_A[i] += (d_B[i*n+k] * d_C[k]);
		}
	}
}


void matDotP(float *h_A, float **h_B, float *h_C, int n) {
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	
	float *h_nB = new float[n*n];
	toOneArray(h_nB, h_B, n);
	
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, n*size);
	cudaMalloc((void **) &d_C, size);
	
	cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_nB, n*size, cudaMemcpyHostToDevice);
	
	dim3 dimGrid( ceil(n / 256.0), 1, 1);
	dim3 dimBlock(256, 1, 1);
	matAddKernel<<< dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
	
	cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main(int argc, char *argv[])
{
	const int n = 100;
	
	float *h_A = new float[n];
	float **h_B = new float*[n];
	float *h_C = new float[n];
	
	for(int i=0; i<n; i++){
		h_B[i] = new float[n];
	}
	
	vectInit(h_C, n);
	matrixInit(h_B, n);
	
	matDotP(h_A, h_B, h_C, n);
	
	printVect(h_A, n);
	
	return 0;
}