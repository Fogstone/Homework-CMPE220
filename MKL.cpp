#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>

void custom_matrix_vector_multiplication(float *matrix, float *vector, int rows, int cols, float *result)
{
    cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, matrix, cols, vector, 1, 0.0f, result, 1);
}

void custom_vector_addition(float *v1, float *v2, int size, float *result)
{
    for (int i = 0; i < size; i++)
    {
        result[i] = v1[i] + v2[i];
    }
}

int main(int argc, char *argv[])
{
    int rows = 3;
    int cols = 2;

    float *matrix = new float[rows * cols];
    float *vector = new float[cols];
    float *output = new float[rows];

    for (int i = 0; i < rows * cols; i++)
    {
        matrix[i] = drand48();
    }
    for (int i = 0; i < cols; i++)
    {
        vector[i] = drand48();
    }

    float bias[rows] = {0.1f, 0.2f, 0.3f};

    custom_matrix_vector_multiplication(matrix, vector, rows, cols, output);
    custom_vector_addition(output, bias, rows, output);

    printf("Result after matrix-vector multiplication and bias addition:\n");
    for (int i = 0; i < rows; i++)
    {
        printf("%.2f\n", output[i]);
    }

    delete[] matrix;
    delete[] vector;
    delete[] output;

    return 0;
}
