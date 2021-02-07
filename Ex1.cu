#include <stdio.h>
#include <time.h>

#define ROUND	32768*32768 // 32k ^ 2 = 1073741824

__global__ void outputFromGPU()
{
	for(int i = 0; i < ROUND; i++){} // GPU
}

int main(void)
{
	printf(":: Ex1 ::\n");

	clock_t begin, end;
	float timeGPU, timeCPU;

	begin = clock();
	outputFromGPU<<<1,1>>>();
	cudaDeviceSynchronize();
	end = clock();
	timeGPU = (float)(end-begin)/CLOCKS_PER_SEC;

	begin = clock();
	for(int i = 0; i < ROUND; i++){} // CPU
	end = clock();
	timeCPU = (float)(end-begin)/CLOCKS_PER_SEC;

	printf("(GPU) : time %f sec.\n", timeGPU);
	printf("(CPU) : time %f sec.\n", timeCPU);

	return 0;
}
