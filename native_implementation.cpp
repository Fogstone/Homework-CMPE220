% % writefile native_example.cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

    void
    custom_matrix_multiplication(float *matrix, float *vector, int rows, int cols, float *result)
{
    for (int i = 0; i < rows; i++)
    {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++)
        {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}

void custom_vector_addition(float *v1, float *v2, int size, float *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = v1[i] + v2[i];
    }
}

double measure_execution_time(void (*func)(float *, float *, int, int, float *), float *matrix, float *vector, int rows, int cols, float *result)
{
    using namespace std::chrono;

    auto start = high_resolution_clock::now();
    func(matrix, vector, rows, cols, result);
    auto end = high_resolution_clock::now();

    duration<double> elapsed = duration_cast<duration<double>>(end - start);
    return elapsed.count();
}

void print_performance_metrics(double seconds, int n)
{
    double Gflops = 2e-9 * n * n * n / seconds;
    printf("%g milliseconds\n", seconds * 1e3);
    printf("Speed:\tn = %d, %.3f Gflop/s\n", n, Gflops);
}

int main(int argc, char *argv[])
{
    int rows = 10000;
    int cols = 20000;

    float *matrix = (float *)malloc(rows * cols * sizeof(float));
    float *vector = (float *)malloc(cols * sizeof(float));
    float *output = (float *)malloc(rows * sizeof(float));

    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = drand48();
    }
    for (int i = 0; i < cols; i++)
    {
        vector[i] = drand48();
    }

    float bias[rows] = {0.1f, 0.2f, 0.3f};

    double time_matvec4 = measure_execution_time(custom_matrix_multiplication, matrix, vector, rows, cols, output);

    printf("[%.2f, %.2f]\n", output[0], output[1]);
    printf("[%.2f, %.2f]\n", output[2], output[3]);

    printf("Matrix-vector multiplication performance:\n");
    print_performance_metrics(time_matvec4, rows);

    free(matrix);
    free(vector);
    free(output);

    return 0;
}
