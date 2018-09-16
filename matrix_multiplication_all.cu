#include "common.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

using namespace std;

#define matrixSize 1000

// Multiply Matrices in GPU
__global__ void multMatrixGPU(long *MatA, long *MatB, long *MatC, int nx, int ny)
{
	unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; // col
	unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y; // row
	unsigned int idx = iy * nx + ix;

	long sum = 0;
	if (ix < nx && iy < ny) {
		for (int i = 0; i < nx; i++) {
			sum += MatA[iy * nx + i] * MatB[i * ny + ix];
		}
		MatC[idx] = sum;
	}
}

// Multiply Matrix in CPU
void multMatrixCPU(long *A, long *B, long *C, const int nx, const int ny)
{
	long *ia = A;
	long *ib = B;
	long *ic = C;
	long sum = 0;

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < nx; j++) {
			for (int k = 0; k < nx; k++) {
				ic[i * nx + j] += ia[i * nx + k] * ib[j + k * nx];
			}
		}
	}

	return;
}

// Multiply Matrix in CPU Parallel
void multMatrixCPUParallel(long *A, long *B, long *C, const int nx, const int ny)
{
	long *ia = A;
	long *ib = B;
	long *ic = C;
	long sum = 0;

	int i, j, k;

	int nProcessors = omp_get_max_threads();

	std::cout << "CPU processors available: " << nProcessors << std::endl;

	omp_set_num_threads(nProcessors/2);

	#pragma omp parallel for private(sum,i,j,k) shared(ia, ib, ic)
	for (i = 0; i < nx; i++) {
		for (j = 0; j < nx; j++) {
			sum = 0;
			for (k = 0; k < nx; k++) {
				sum += ia[i * nx + k] * ib[k * nx + j];
			}
			ic[i * nx + j] = sum;
		}
	}

	return;
}

// Fill Matrix with random ints
void initialData(long *ip, const int size)
{
	int i;

	for (i = 0; i < size; i++)
	{
		ip[i] = rand() % 10;
	}

	return;
}

// Check if matrices match
void checkResult(long *hostRef, long *gpuRef, const int N)
{
	// double epsilon = 1.0E-8;
	bool match = 1;

	for (int i = 0; i < N; i++)
	{
		if (hostRef[i] != gpuRef[i])
		{
			match = 0;
			printf("host %f gpu %f\n", hostRef[i], gpuRef[i]);
			break;
		}
	}

	if (match)
		printf("Arrays match.\n\n");
	else
		printf("Arrays do not match.\n\n");
}

// Print Matrix (for debug)
void printMatrix(long *A, const int nx, const int ny)
{
	long *ia = A;

	for (int iy = 0; iy < ny; iy++)
	{
		for (int ix = 0; ix < nx; ix++)
		{
			printf("%d     ", ia[ix]);
		}
		printf("\n");
		ia += nx;
	}

	return;
}

int main(int argc, char **argv)
{
	printf("%s Starting...\n", argv[0]);

	// set up device
	int dev = 0;
	cudaDeviceProp deviceProp;
	SAFE_CALL(cudaGetDeviceProperties(&deviceProp, dev), "Error device prop");
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	SAFE_CALL(cudaSetDevice(dev), "Error setting device");

	// set up data size of matrix
	int nx = matrixSize;
	int ny = matrixSize;

	int nxy = nx * ny;
	int nBytes = nxy * sizeof(long);
	printf("Matrix size: nx %d ny %d\n", nx, ny);

	// malloc host memory
	long *h_A, *h_B, *hostRef, *gpuRef, *hostRefParallel;
	h_A = (long *)malloc(nBytes);
	h_B = (long *)malloc(nBytes);
	hostRef = (long *)malloc(nBytes);
	gpuRef = (long *)malloc(nBytes);
	hostRefParallel = (long *)malloc(nBytes);

	// initialize data at host side

	initialData(h_A, nxy);
	initialData(h_B, nxy);

	memset(hostRef, 0, nBytes);
	memset(gpuRef, 0, nBytes);

	// malloc device global memory
	long *d_MatA, *d_MatB, *d_MatC;
	SAFE_CALL(cudaMalloc((void **)&d_MatA, nBytes), "Error allocating d_MatA");
	SAFE_CALL(cudaMalloc((void **)&d_MatB, nBytes), "Error allocating d_MatB");
	SAFE_CALL(cudaMalloc((void **)&d_MatC, nBytes), "Error allocating d_MatC");

	// transfer data from host to device
	SAFE_CALL(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatA");
	SAFE_CALL(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice), "Error copying d_MatB");

	// invoke kernel at host side
	dim3 block(8, 16);

	dim3 grid(256, 256);
	//dim3 grid((nx + block.x - 1) / block.x, (nx + block.y - 1) / block.y);

	// GPU multiplication
	auto start_cpu = chrono::high_resolution_clock::now();
	multMatrixGPU << <grid, block >> > (d_MatA, d_MatB, d_MatC, nx, ny);
	SAFE_CALL(cudaDeviceSynchronize(), "Error executing kernel");
	auto end_cpu = chrono::high_resolution_clock::now();
	chrono::duration<float, std::milli> duration_ms = end_cpu - start_cpu;

	printf("multMatrixGPU <<<(%d,%d), (%d,%d)>>> elapsed %f ms\n", grid.x, grid.y, block.x, block.y, duration_ms.count());

	// SAFE_CALL kernel error
	SAFE_CALL(cudaGetLastError(), "Error with last error");

	// copy kernel result back to host side
	SAFE_CALL(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost), "Error copying d_MatC");


	// CPU parallel multiplication
	start_cpu = chrono::high_resolution_clock::now();
	multMatrixCPUParallel(h_A, h_B, hostRefParallel, nx, ny);
	end_cpu = chrono::high_resolution_clock::now();
	duration_ms = end_cpu - start_cpu;

	printf("multMatrixCPUParallel elapsed %f ms\n", duration_ms.count());

	// check CPU parallel against GPU results
	cout << "\n" << "\n";
	cout << "check CPU parallel against GPU results: ";
	checkResult(hostRefParallel, gpuRef, nxy);

	// CPU multiplication
	start_cpu = chrono::high_resolution_clock::now();
	multMatrixCPU(h_A, h_B, hostRef, nx, ny);
	end_cpu = chrono::high_resolution_clock::now();
	duration_ms = end_cpu - start_cpu;

	printf("multMatrixCPU elapsed %f ms\n", duration_ms.count());

	cout << "\n" << "\n";

	// check CPU against GPU results
	cout << "check CPU against GPU results: ";
	checkResult(hostRef, gpuRef, nxy);

	// check both CPU results
	cout << "check CPU parallel against CPU results: ";
	checkResult(hostRef, hostRefParallel, nxy);

	// Print Matrices
	// printMatrix(hostRef, nx, ny);
	// printf("\n");
	// printMatrix(gpuRef, nx, ny);
	// printf("\n");
	// printMatrix(hostRefParallel, nx, ny);

	// free device global memory
	SAFE_CALL(cudaFree(d_MatA), "Error freeing memory");
	SAFE_CALL(cudaFree(d_MatB), "Error freeing memory");
	SAFE_CALL(cudaFree(d_MatC), "Error freeing memory");

	// free host memory
	free(h_A);
	free(h_B);
	free(hostRef);
	free(gpuRef);
	free(hostRefParallel);

	// reset device
	SAFE_CALL(cudaDeviceReset(), "Error reseting");

	return (0);
}
