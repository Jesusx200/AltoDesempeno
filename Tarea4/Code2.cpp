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

void printVec(float **h_x, int n) {
	cout << "[ ";
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < n; j++) {
			cout << h_x[i][j] << " ";
		}
		cout << "| \n";
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
	
	if(i < (n*n) && (i%n == 0)) {
		for(int k=0; k<n; k++){
			d_C[k+i] = d_A[k+i] + d_B[k+i];
		}
	}
}

void matAdd(float **h_A, float **h_B, float **h_C, int n) {
	int size = n*n * sizeof(float);
	float *d_A, *d_B, *d_C;
	float *h_nA = new float[n*n];
	float *h_nB = new float[n*n];
	float *h_nC = new float[n*n];
	
	toOneArray(h_nA, h_A, n);
	toOneArray(h_nB, h_B, n);
	toOneArray(h_nC, h_C, n);
	
	cudaMalloc((void **) &d_A, size);
	cudaMalloc((void **) &d_B, size);
	cudaMalloc((void **) &d_C, size);
	
	cudaMemcpy(d_A, h_nA, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_nB, size, cudaMemcpyHostToDevice);
	
	dim3 dimGrid( ceil(n*n / 256.0), 1, 1);
	dim3 dimBlock(256, 1, 1);
	matAddKernel<<< dimGrid, dimBlock>>>(d_A, d_B, d_C, n);
	
	cudaMemcpy(h_nC, d_C, size, cudaMemcpyDeviceToHost);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	
	toArrayFromOne(h_nC, h_C, n);
}

int main(int argc, char *argv[])
{
	const int n = 100;
	
	float **h_A = new float*[n];
	float **h_B = new float*[n];
	float **h_C = new float*[n];
	
	for(int i=0; i<n; i++){
		h_A[i] = new float[n];
		h_B[i] = new float[n];
		h_C[i] = new float[n];
	}
	
	matrixInit(h_A, n);
	matrixInit(h_B, n);
	
	matAdd(h_A, h_B, h_C, n);
	
	printVec(h_C, n);
	
	return 0;
}
