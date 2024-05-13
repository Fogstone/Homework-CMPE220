#include <cassert>
#include <cstdlib>
#include <iostream>
#include <chrono>

#define CUSTOM_MASK_LENGTH 7

__constant__ int custom_mask[CUSTOM_MASK_LENGTH];

__global__ void custom_convolution_1d(int *input_array, int *output_result, int array_size)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ int shared_array[];
    shared_array[threadIdx.x] = input_array[thread_id];
    __syncthreads();

    int temp = 0;

    for (int j = 0; j < CUSTOM_MASK_LENGTH; j++)
    {
        if ((threadIdx.x + j) >= blockDim.x)
        {
            temp += input_array[thread_id + j] * custom_mask[j];
        }
        else
        {
            temp += shared_array[threadIdx.x + j] * custom_mask[j];
        }
    }

    output_result[thread_id] = temp;
}

void verify_custom_result(int *input_array, int *mask, int *output_result, int array_size)
{
    int temp;
    for (int i = 0; i < array_size; i++)
    {
        temp = 0;
        for (int j = 0; j < CUSTOM_MASK_LENGTH; j++)
        {
            temp += input_array[i + j] * mask[j];
        }
        assert(temp == output_result[i]);
    }
}

int main()
{
    int array_size = 1048576;
    int bytes_array = array_size * sizeof(int);
    size_t bytes_mask = CUSTOM_MASK_LENGTH * sizeof(int);
    int radius = CUSTOM_MASK_LENGTH / 2;
    int padded_array_size = array_size + radius * 2;
    size_t bytes_padded_array = padded_array_size * sizeof(int);
    int *input_array = new int[padded_array_size];
    for (int i = 0; i < padded_array_size; i++)
    {
        if ((i < radius) || (i >= (array_size + radius)))
        {
            input_array[i] = 0;
        }
        else
        {
            input_array[i] = rand() % 100;
        }
    }
    int *custom_mask_values = new int[CUSTOM_MASK_LENGTH];
    for (int i = 0; i < CUSTOM_MASK_LENGTH; i++)
    {
        custom_mask_values[i] = rand() % 10;
    }
    int *output_result = new int[array_size];
    int *d_input_array, *d_output_result;
    cudaMalloc(&d_input_array, bytes_padded_array);
    cudaMalloc(&d_output_result, bytes_array);
    cudaMemcpy(d_input_array, input_array, bytes_padded_array, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(custom_mask, custom_mask_values, bytes_mask);

    int THREADS = 256;
    int GRID = (array_size + THREADS - 1) / THREADS;
    size_t SHARED_MEMORY = THREADS * sizeof(int);

    auto start_time = std::chrono::high_resolution_clock::now();
    custom_convolution_1d<<<GRID, THREADS, SHARED_MEMORY>>>(d_input_array, d_output_result, array_size);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double execution_time = elapsed_seconds.count();
    long long total_flops = static_cast<long long>(array_size) * CUSTOM_MASK_LENGTH;
    double gflops = total_flops / (execution_time * 1e9);
    std::cout << "GFLOPS: " << gflops << std::endl;

    cudaMemcpy(output_result, d_output_result, bytes_array, cudaMemcpyDeviceToHost);
    verify_custom_result(input_array, custom_mask_values, output_result, array_size);
    delete[] input_array;
    delete[] output_result;
    delete[] custom_mask_values;
    cudaFree(d_input_array);
    cudaFree(d_output_result);
    return 0;
}
