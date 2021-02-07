#include <stdio.h>
#include <time.h>

#define N 4

__global__ void outputFromGPU()
{
	printf("[%d] : [%d]\n", blockIdx.x, threadIdx.x);
}

__global__ void multiplicationTableBlock(int *mutex, int *index)
{
	int c = blockIdx.x;

	while(atomicExch(mutex, 1) != 0);
	for(int i = 1; i <= 12; i++)
	{
		printf("[%d]\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d\n",
				c, *index, i, *index*i, *index+1, i, (*index+1)*i, *index+2, i, (*index+2)*i, *index+3, i, (*index+3)*i);
	}
	printf("\n");
	if(*index == 10){*index = 2;}
	*index += 4;
	atomicExch(mutex, 0);
}

__global__ void multiplicationTableThread()
{
	int c = threadIdx.x;
	int i = c+1;

	for(int index = 2; index <= 10; index+=4)
	{
		printf("[%d]\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d\n",
				c, index, i, index*i, index+1, i, (index+1)*i, index+2, i, (index+2)*i, index+3, i, (index+3)*i);
		if(i == 12){printf("\n");}
	}
}

__global__ void multiplicationTableBlockAndThread(int *mutex, int *index)
{
	int c = blockIdx.x;
	int i = threadIdx.x+1;
	
	if(threadIdx.x == 0)
	{
		while(atomicExch(mutex, 1) != 0);
		*index += 4;
		atomicExch(mutex, 0);
	}

	printf("[%d] : [%d]\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d\n%s",
	c, threadIdx.x, *index, i, *index*i, *index+1, i, (*index+1)*i, *index+2, i, (*index+2)*i, *index+3, i, (*index+3)*i, (i == 12)?"\n":"");
}

int main(void)
{
	printf(":: Ex3 ::\n\n");

	int *h_index, *d_index, *d_mutex;
	h_index = (int*)malloc(sizeof(int));
	cudaMalloc((void**)&d_index, sizeof(int));
	cudaMalloc((void**)&d_mutex, sizeof(int));

	printf("\n::: Block only :::\n");
	outputFromGPU<<<N, 1>>>();
	cudaDeviceSynchronize();

	printf("\n::: Thread only :::\n");
	outputFromGPU<<<1, N>>>();
	cudaDeviceSynchronize();

	printf("\n::: Block and Thread :::\n");
	outputFromGPU<<<N, N>>>();
	cudaDeviceSynchronize();

	printf("\n::: Multiplication Table Block :::\n\n");
	*h_index = 2;
	cudaMemcpy(d_index, h_index, sizeof(int), cudaMemcpyHostToDevice);
	multiplicationTableBlock<<<3, 1>>>(d_mutex, d_index);
	cudaDeviceSynchronize();

	printf("::: Multiplication Table Thread :::\n\n");
	multiplicationTableThread<<<1, 12>>>();
	cudaDeviceSynchronize();

	printf("::: Multiplication Table Block and Thread :::\n\n");
	*h_index = -2;
	cudaMemcpy(d_index, h_index, sizeof(int), cudaMemcpyHostToDevice);
	multiplicationTableBlockAndThread<<<3, 12>>>(d_mutex, d_index);
	cudaDeviceSynchronize();

	printf("::: Multiplication Table CPU :::\n\n");

	for(int i = 2; i <= 10; i+=4)
	{
		for(int a = 1; a <= 12; a++)
		{
			printf("%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d \t\t%d x %d = %d\n",
					i, a, i*a, i+1, a, (i+1)*a, i+2, a, (i+2)*a, i+3, a, (i+3)*a);
		}
		printf("\n");
	}

	cudaFree(d_index);
	cudaFree(d_mutex);
	free(h_index);

	return 0;
}
