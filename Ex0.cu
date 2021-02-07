#include <stdio.h>

__global__ void outputFromGPU()
{
	printf("Hello World!!! from GPU.\n");
}

int main(void)
{
	printf(":: Ex0 ::\n");

	outputFromGPU<<<1,1>>>();

	printf("Hello World!!! from CPU.\n");

	return 0;
}
