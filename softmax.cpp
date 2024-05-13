% % writefile Softmax.cpp
#include <iostream>
#include <numeric>
#include <chrono>
#include <cmath>

    template <typename Type>
    void custom_softmax(Type *data, size_t size)
{
    Type max_value = *std::max(data, data + size);
    for (size_t i = 0; i < size; ++i)
    {
        data[i] = std::exp(data[i] - max_value);
    }

    Type sum_exp = std::accumulate(data, data + size, Type(0));
    for (size_t i = 0; i < size; ++i)
    {
        data[i] /= sum_exp;
    }
}

double custom_benchmark_softmax(size_t size)
{
    double *input_data = new double[size];
    for (size_t i = 0; i < size; ++i)
    {
        input_data[i] = static_cast<double>(rand()) / RAND_MAX;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    custom_softmax(input_data, size);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    delete[] input_data;
    return elapsed_seconds.count();
}

int main()
{
    size_t array_size = 1048576;
    double execution_time = custom_benchmark_softmax(array_size);
    double gflops = 2.0 * array_size * std::log(array_size) / (execution_time * 1e9);
    std::cout << "Custom Softmax Benchmark (size " << array_size << "):" << std::endl;
    std::cout << "Execution Time (s): " << execution_time << std::endl;
    std::cout << "GFLOPs: " << gflops << std::endl;
    return 0;
}
