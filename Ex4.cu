#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 22528*22528 // 22k ^ 2 = 507510784
#define THREAD_PER_BLOCK 1024
#define GRID N/THREAD_PER_BLOCK

__global__ void outputFromGPU(int *arr, int *min, int *max, int *sum, int *mutex)
{
	//GPU
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = gridDim.x * blockDim.x;
	int offset = stride;

	__shared__ int s_min[THREAD_PER_BLOCK];
	__shared__ int s_max[THREAD_PER_BLOCK];
	__shared__ int s_sum[THREAD_PER_BLOCK];

	int _min = arr[index];
	int _max = arr[index];
	int _sum = arr[index];

	while(index + offset < N)
	{
		_min = (_min > arr[index + offset])?arr[index + offset]:_min;
		_max = (_max < arr[index + offset])?arr[index + offset]:_max;
		_sum += arr[index + offset];
		offset += stride;
	}

	s_min[threadIdx.x] = _min;
	s_max[threadIdx.x] = _max;
	s_sum[threadIdx.x] = _sum;

	__syncthreads();

	int i = blockDim.x / 2;
	while(i != 0)
	{
		if(threadIdx.x < i)
		{
			s_min[threadIdx.x] = (s_min[threadIdx.x] > s_min[threadIdx.x + i])?s_min[threadIdx.x + i]:s_min[threadIdx.x];
			s_max[threadIdx.x] = (s_max[threadIdx.x] < s_max[threadIdx.x + i])?s_max[threadIdx.x + i]:s_max[threadIdx.x];
			s_sum[threadIdx.x] += s_sum[threadIdx.x + i];
		}

		__syncthreads();
		i /= 2;
	}

	if(threadIdx.x == 0)
	{
		while(atomicCAS(mutex, 0, 1) != 0);
		*min = (*min > s_min[0])?s_min[0]:*min;
		*max = (*max < s_max[0])?s_max[0]:*max;
		*sum += s_sum[0];
		atomicExch(mutex, 0);
	}
}

__host__ void get_min_max(int *arr, int *min, int *max, int *sum)
{
	//CPU
	*min = arr[0];
	*max = arr[0];
	*sum = arr[0];

	for(int i = 1; i < N; i++)
	{
			*min = (*min > arr[i])?*min = arr[i]:*min;
			*max = (*max < arr[i])?*max = arr[i]:*max;
			*sum += arr[i];
	}
}

__host__ void get_min_max_advance(int *arr, int *min, int *max, int *sum)
{
	//CPU
	int t;
	int stride = N/2;

	t = (arr[0] > arr[stride])?arr[stride]:arr[0];
	*min = (*min > t)?*min = t:*min;
	t = (arr[0] < arr[stride])?arr[stride]:arr[0];
	*max = (*max < t)?*max = t:*max;
	*sum = arr[0]+arr[stride];

	for(int i = 1; i < stride; i++)
	{
			t = (arr[i] > arr[i+stride])?arr[i+stride]:arr[i];
			*min = (*min > t)?*min = t:*min;
			t = (arr[i] < arr[i+stride])?arr[i+stride]:arr[i];
			*max = (*max < t)?*max = t:*max;
			*sum += arr[i]+arr[i+stride];
	}
}

int main(void)
{
	printf(":: Ex4 ::\n\n");

	int *arr, *d_arr, *h_min, *h_max, *h_sum, *d_min, *d_max, *d_sum, *d_mutex;
	int i, min, max, sum;
	clock_t begin, end;
	float timeGPU, timeCPU;
	int size = sizeof(int);

	arr = (int*)malloc(size * N);
	h_min = (int*)malloc(size);
	h_max = (int*)malloc(size);
	h_sum = (int*)malloc(size);
	cudaMalloc((void**)&d_arr, size * N);
	cudaMalloc((void**)&d_min, size);
	cudaMalloc((void**)&d_max, size);
	cudaMalloc((void**)&d_sum, size);
	cudaMalloc((void**)&d_mutex, size);

	printf("Initializing... please wait.\n\n");

	srand (time(NULL));
	for(i = 0; i < N; i++){arr[i] = rand()%20000;}

	printf("GPU Parallel Algorithm...\n");

	*h_min = 20000;
	*h_max = 0;
	*h_sum = 0;

	begin = clock();
	cudaMemcpy(d_arr, arr, size * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_min, h_min, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_max, h_max, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_sum, h_sum, size, cudaMemcpyHostToDevice);
	outputFromGPU<<<GRID, THREAD_PER_BLOCK>>>(d_arr, d_min, d_max, d_sum, d_mutex);
	cudaDeviceSynchronize();
	cudaMemcpy(h_min, d_min, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_max, d_max, size, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);
	end = clock();
	timeGPU = (float)(end-begin)/CLOCKS_PER_SEC;

	printf("(GPU) : time %f sec. : min: %d, max: %d, sum: %d\n\n", timeGPU, *h_min, *h_max, *h_sum);
	printf("CPU Basic Algorithm...\n");

	begin = clock();
	get_min_max(arr, &min, &max, &sum);
	end = clock();
	timeCPU = (float)(end-begin)/CLOCKS_PER_SEC;

	printf("(CPU) : time %f sec. : min: %d, max: %d, sum: %d\n\n", timeCPU, min, max, sum);
	printf("CPU Advance Algorithm...\n");

	begin = clock();
	get_min_max_advance(arr, &min, &max, &sum);
	end = clock();
	timeCPU = (float)(end-begin)/CLOCKS_PER_SEC;

	printf("(CPU) : time %f sec. : min: %d, max: %d, sum: %d\n", timeCPU, min, max, sum);

	cudaFree(d_arr);
	cudaFree(d_min);
	cudaFree(d_max);
	cudaFree(d_sum);
	cudaFree(d_mutex);
	free(arr);
	free(h_min);
	free(h_max);
	free(h_sum);


	return 0;
}
