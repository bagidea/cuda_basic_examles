#include <stdio.h>
#include <time.h>

#define ROUND	32768*32768 // 32k ^ 2 = 1073741824

__global__ void outputFromGPU(int *cg)
{
	for(int i = 0; i < ROUND; i++){ *cg += 1; } // GPU
}

int main(void)
{
	printf(":: Ex2 ::\n");

	int cg, c;
	int *d_cg;
	clock_t begin, end;
	float timeGPU, timeCPU;

	cg = 0;
	begin = clock();
	cudaMalloc((void**)&d_cg, sizeof(int));
	cudaMemcpy(d_cg, &cg, sizeof(int), cudaMemcpyHostToDevice);
	outputFromGPU<<<1,1>>>(d_cg);
	cudaDeviceSynchronize();
	cudaMemcpy(&cg, d_cg, sizeof(int), cudaMemcpyDeviceToHost);
	end = clock();
	timeGPU = (float)(end-begin)/CLOCKS_PER_SEC;
	cudaFree(d_cg);

	c = 0;
	begin = clock();
	for(int i = 0; i < ROUND; i++){ c++; } // CPU
	end = clock();
	timeCPU = (float)(end-begin)/CLOCKS_PER_SEC;

	printf("(GPU) : time %f sec. : Value = %d\n", timeGPU, cg);
	printf("(CPU) : time %f sec. : Value = %d\n", timeCPU, c);

	return 0;
}
