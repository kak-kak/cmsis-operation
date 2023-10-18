#include <stdio.h>
#include <arm_math.h>
#include <assert.h>
// #include "arm_math.h"  // Include CMSIS-DSP library
#include <string.h>  // For memset

void custom_conv1d(
    float32_t* input,
    float32_t* kernel,
    float32_t* bias,
    float32_t* output,
    uint32_t input_channels,
    uint32_t output_channels,
    uint32_t input_size,
    uint32_t kernel_size
) {
    uint32_t output_size = input_size - kernel_size + 1;
    memset(output, 0, output_channels * output_size * sizeof(float32_t)); 

    for (uint32_t oc = 0; oc < output_channels; ++oc) {
        for (uint32_t ic = 0; ic < input_channels; ++ic) {
            for (uint32_t i = 0; i < output_size; ++i) {
                float32_t sum = 0;
                for (uint32_t j = 0; j < kernel_size; ++j) {
                    sum += input[ic * input_size + i + j] * kernel[oc * input_channels * kernel_size + ic * kernel_size + j];
                }
                output[oc * output_size + i] += sum;
            }
        }
        for (uint32_t i = 0; i < output_size; ++i) {
            output[oc * output_size + i] += bias[oc];
        }
    }
}


void custom_linear(const float32_t *input, const float32_t *weight, const float32_t *bias, float32_t *output, const uint16_t num_rows, const uint16_t input_dim, const uint16_t output_dim) {
    // Create CMSIS-DSP matrix structures
    arm_matrix_instance_f32 mat_input;
    arm_matrix_instance_f32 mat_weight;
    arm_matrix_instance_f32 mat_output;

    // Initialize CMSIS-DSP matrix structures
    arm_mat_init_f32(&mat_input, num_rows, input_dim, (float32_t *)input);
    arm_mat_init_f32(&mat_weight, output_dim, input_dim, (float32_t *)weight);  // Note: dimensions are swapped
    arm_matrix_instance_f32 mat_weight_transposed;
    float32_t weight_transposed[output_dim * input_dim];
    arm_mat_init_f32(&mat_weight_transposed, input_dim, output_dim, weight_transposed);
    arm_mat_trans_f32(&mat_weight, &mat_weight_transposed);  // Transpose the weight matrix
    arm_mat_init_f32(&mat_output, num_rows, output_dim, output);

    // Perform matrix multiplication: output = input * weight^T
    arm_mat_mult_f32(&mat_input, &mat_weight_transposed, &mat_output);

    // Add bias to each element of the output matrix
    for (uint16_t i = 0; i < num_rows; ++i) {
        for (uint16_t j = 0; j < output_dim; ++j) {
            output[i * output_dim + j] += bias[j];
        }
    }
}



// import torch
// import torch.nn.functional as F
// import numpy as np

// # Input, kernel, and bias data
// input_data = torch.ones(1, 3, 20)
// kernel_data = torch.from_numpy(np.arange(30).reshape(2, 3, 5).astype(np.float32))
// bias_data = torch.ones(2)

// # Compute the convolution
// output = F.conv1d(input_data, kernel_data, bias_data)

// # Print the output
// print(output)
void test_custom_conv1d() {
    float32_t input[1 * 3 * 20];
    float32_t kernel[2 * 3 * 5];
    float32_t bias[2];
    float32_t output[2 * (20 - 5 + 1)];

    for (int i = 0; i < 1 * 3 * 20; ++i) {
        input[i] = 1.0f;
    }

    for (int i = 0; i < 2 * 3 * 5; ++i) {
        kernel[i] = (float32_t)i;
    }

    for (int i = 0; i < 2; ++i) {
        bias[i] = 1.0f;
    }

    custom_conv1d(input, kernel, bias, output, 3, 2, 20, 5);

    float32_t expected_output[2][16] = {
        {106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106, 106},
        {331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331, 331}
    };

    for (int oc = 0; oc < 2; ++oc) {
        for (int i = 0; i < (20 - 5 + 1); ++i) {
            assert(output[oc * (20 - 5 + 1) + i] == expected_output[oc][i]);
        }
    }
    printf("conv1d test passed\n");
}


// import torch

// # Define the input, weight, and bias tensors
// input_tensor = torch.tensor([
//     [1.0, 2.0, 3.0, 4.0, 5.0],
//     [6.0, 7.0, 8.0, 9.0, 10.0],
//     [11.0, 12.0, 13.0, 14.0, 15.0]
// ])
// weight_tensor = torch.tensor([
//     [0.1, 0.5, 0.9, 1.3, 1.7],
//     [0.2, 0.6, 1.0, 1.4, 1.8],
//     [0.3, 0.7, 1.1, 1.5, 1.9],
//     [0.4, 0.8, 1.2, 1.6, 2.0]
// ])
// bias_tensor = torch.tensor([0.1, 0.2, 0.3, 0.4])

// # Perform the linear transformation
// output_tensor = torch.nn.functional.linear(input_tensor, weight_tensor, bias_tensor)

// # Print the reference output values to use in the C code
// print(output_tensor.numpy())
void test_custom_liner(){
    float32_t input[3][5] = {
        {1.0f, 2.0f, 3.0f, 4.0f, 5.0f},
        {6.0f, 7.0f, 8.0f, 9.0f, 10.0f},
        {11.0f, 12.0f, 13.0f, 14.0f, 15.0f}
    };
    float32_t weight[4][5] = {
        {0.1, 0.5, 0.9, 1.3, 1.7},
        {0.2, 0.6, 1.0, 1.4, 1.8},
        {0.3, 0.7, 1.1, 1.5, 1.9},
        {0.4, 0.8, 1.2, 1.6, 2.0}
    };

    float32_t bias[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    float32_t output[3][4] = {0};

    // Define the reference output values obtained from PyTorch
    float32_t expected_output[3][4] = {
        {17.6, 19.2, 20.8, 22.4},
        {40.1, 44.2, 48.3, 52.4},
        {62.6, 69.2, 75.8, 82.4}
    };

    // Call the custom_linear function
    custom_linear((float32_t *)input, (float32_t *)weight, bias, (float32_t *)output, 3, 5, 4);

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            assert(fabs(output[i][j] - expected_output[i][j]) < 1e-6);
        }
    }
    printf("liner test passed\n");
}

int main() {
    test_custom_conv1d();
    test_custom_liner();
    return 0;
}
