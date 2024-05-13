#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <vector>
#include <chrono>

using std::cout;
using std::generate;
using std::vector;

__global__ void custom_matrix_Mult(const int *a, const int *b, int *c, int *d, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = 0;
    for (int i = 0; i < N; ++i)
    {
        c[idx] += a[idx * N + i] * b[i];
    }
    c[idx] += d[idx];
}

void verify_result(vector<int> &a, vector<int> &b, vector<int> &c, vector<int> &d, int N)
{
    for (int i = 0; i < N; ++i)
    {
        int tmp = 0;
        for (int j = 0; j < N; ++j)
        {
            tmp += a[i * N + j] * b[j];
        }
        tmp += d[i];
        assert(tmp == c[i]);
    }
}

int main()
{
    int N = 1000;
    size_t matrix_bytes = N * N * sizeof(int);
    size_t vector_bytes = N * sizeof(int);
    vector<int> h_a(N * N);
    vector<int> h_b(N);
    vector<int> h_c(N);
    vector<int> h_d(N);

    generate(h_a.begin(), h_a.end(), []()
             { return rand() % 100; });
    generate(h_b.begin(), h_b.end(), []()
             { return rand() % 100; });
    generate(h_d.begin(), h_d.end(), []()
             { return rand() % 100; });

    int *d_a, *d_b, *d_c, *d_d;
    cudaMalloc(&d_a, matrix_bytes);
    cudaMalloc(&d_b, vector_bytes);
    cudaMalloc(&d_c, vector_bytes);
    cudaMalloc(&d_d, vector_bytes);

    cudaMemcpy(d_a, h_a.data(), matrix_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), vector_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_d, h_d.data(), vector_bytes, cudaMemcpyHostToDevice);

    int THREADS = 1024;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    auto start_time = std::chrono::high_resolution_clock::now();
    custom_matrix_Mult<<<BLOCKS, THREADS>>>(d_a, d_b, d_c, d_d, N);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    double execution_time = elapsed_seconds.count();

    long long total_flops = 2 * static_cast<long long>(N * N * N);
    double gflops = total_flops / (execution_time * 1e9);

    std::cout << "GFLOPS: " << gflops << std::endl;

    cudaMemcpy(h_c.data(), d_c, vector_bytes, cudaMemcpyDeviceToHost);

    verify_result(h_a, h_b, h_c, h_d, N);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_d);

    return 0;
}
