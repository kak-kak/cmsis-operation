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


// void custom_layer_norm(float32_t* input, const int* input_shape, float32_t* weight, float32_t* bias, float32_t* output) {
//     int batch_size = input_shape[0];
//     int num_channels = input_shape[1];
//     int feature_size = input_shape[2];
    
//     for (int i = 0; i < batch_size; ++i) {
//         for (int j = 0; j < num_channels; ++j) {
//             float32_t mean, var, inv_var, std_dev;
//             float32_t* current_input = input + (i * num_channels + j) * feature_size;
//             float32_t* current_output = output + (i * num_channels + j) * feature_size;
            
//             // Calculate the mean of the current channel
//             arm_mean_f32(current_input, feature_size, &mean);
//             // Calculate the variance of the current channel
//             arm_var_f32(current_input, feature_size, &var);
//             // Calculate the square root of the variance plus epsilon for numerical stability
//             arm_sqrt_f32(var + 1e-05f, &std_dev);
//             // var = var * (feature_size / (feature_size - 1.0f));

//             // Calculate the inverse of the standard deviation
//             inv_var = 1.0f / std_dev;
            
//             // Perform the layer normalization on the current channel
//             for (int k = 0; k < feature_size; ++k) {
//                 current_output[k] = (current_input[k] - mean) * inv_var * weight[k] + bias[k];
//             }
//         }
//     }
// }
void custom_layer_norm(float32_t* input, const int* input_shape, float32_t* weight, float32_t* bias, float32_t* output) {
    int batch_size = input_shape[0];
    int num_channels = input_shape[1];
    int feature_size = input_shape[2];
    
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < num_channels; ++j) {
            float32_t mean = 0.0f, var = 0.0f, inv_var, std_dev;
            float32_t* current_input = input + (i * num_channels + j) * feature_size;
            float32_t* current_output = output + (i * num_channels + j) * feature_size;
            
            // Calculate the mean of the current channel manually
            for (int k = 0; k < feature_size; ++k) {
                mean += current_input[k];
            }
            mean /= feature_size;
            
            // Calculate the variance of the current channel manually
            for (int k = 0; k < feature_size; ++k) {
                float32_t diff = current_input[k] - mean;
                var += diff * diff;
            }
            var /= feature_size;
            
            // Calculate the square root of the variance plus epsilon for numerical stability
            arm_sqrt_f32(var + 1e-05f, &std_dev);
            // Calculate the inverse of the standard deviation
            inv_var = 1.0f / std_dev;
            
            // Perform the layer normalization on the current channel
            for (int k = 0; k < feature_size; ++k) {
                current_output[k] = (current_input[k] - mean) * inv_var * weight[k] + bias[k];
            }
        }
    }
}


void custom_transpose(float input[2][3][5], float output[3][2][5], int dim1, int dim2, int dim3) {
    arm_matrix_instance_f32 input_mat, output_mat;
    
    if (dim1 == 0 && dim2 == 1) {
        for (int i = 0; i < dim3; ++i) {
            float input_slice[2][3];
            float output_slice[3][2];
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 3; ++k) {
                    input_slice[j][k] = input[j][k][i];
                }
            }
            arm_mat_init_f32(&input_mat, 2, 3, (float *)input_slice);
            arm_mat_init_f32(&output_mat, 3, 2, (float *)output_slice);
            arm_mat_trans_f32(&input_mat, &output_mat);
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 2; ++k) {
                    output[j][k][i] = output_slice[j][k];
                }
            }
        }
    } else if (dim1 == 1 && dim2 == 2) {
        for (int i = 0; i < dim1; ++i) {
            float input_slice[3][5];
            float output_slice[5][3];
            for (int j = 0; j < 3; ++j) {
                for (int k = 0; k < 5; ++k) {
                    input_slice[j][k] = input[i][j][k];
                }
            }
            arm_mat_init_f32(&input_mat, 3, 5, (float *)input_slice);
            arm_mat_init_f32(&output_mat, 5, 3, (float *)output_slice);
            arm_mat_trans_f32(&input_mat, &output_mat);
            for (int j = 0; j < 5; ++j) {
                for (int k = 0; k < 3; ++k) {
                    output[i][j][k] = output_slice[j][k];
                }
            }
        }
    } else if (dim1 == 0 && dim2 == 2) {
        for (int i = 0; i < dim2; ++i) {
            float input_slice[2][5];
            float output_slice[5][2];
            for (int j = 0; j < 2; ++j) {
                for (int k = 0; k < 5; ++k) {
                    input_slice[j][k] = input[j][i][k];
                }
            }
            arm_mat_init_f32(&input_mat, 2, 5, (float *)input_slice);
            arm_mat_init_f32(&output_mat, 5, 2, (float *)output_slice);
            arm_mat_trans_f32(&input_mat, &output_mat);
            for (int j = 0; j < 5; ++j) {
                for (int k = 0; k < 2; ++k) {
                    output[k][i][j] = output_slice[j][k];
                }
            }
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


void test_custom_layerNorm() {
        float32_t input[2][3][5] = {
        {
            {0.1, 0.2, 0.3, 0.4, 0.5},
            {0.6, 0.7, 0.8, 0.9, 1.0},
            {1.1, 1.2, 1.3, 1.4, 1.5}
        },
        {
            {1.6, 1.7, 1.8, 1.9, 2.0},
            {2.1, 2.2, 2.3, 2.4, 2.5},
            {2.6, 2.7, 2.8, 2.9, 3.0}
        }
    };
    
    int input_shape[] = {2, 3, 5};  // The dimensions of the input tensor
    int normalized_shape = 5;  // The size of the normalization dimension
    
    // Define the weight and bias vectors with 5 elements each
    float32_t weight[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    float32_t bias[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    
    float32_t output[2][3][5] = {0};  // The output tensor to store the result
    
    // Call the custom_layer_norm function
    custom_layer_norm((float32_t *)input, input_shape, weight, bias, (float32_t *)output);
    
    // Define the expected output tensor based on the output from PyTorch
    float32_t expected_output[2][3][5] = {
        {
            {-1.4139e+00, -7.0693e-01, 3.6589e-08, 7.0693e-01, 1.4139e+00},
            {-1.4139e+00, -7.0693e-01, -2.0183e-07, 7.0693e-01, 1.4139e+00},
            {-1.4139e+00, -7.0693e-01, 2.8280e-07, 7.0693e-01, 1.4139e+00}
        },
        {
            {-1.4139e+00, -7.0693e-01, -4.3246e-07, 7.0693e-01, 1.4139e+00},
            {-1.4139e+00, -7.0693e-01, -6.2319e-07, 7.0693e-01, 1.4139e+00},
            {-1.4139e+00, -7.0693e-01, 8.0732e-07, 7.0693e-01, 1.4139e+00}
        }
    };
    
    // Define a small tolerance for floating-point comparisons
    float32_t tolerance = 1e-3;
    
    // Compare the output tensor from custom_layer_norm with the expected output tensor
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 5; ++k) {
                assert(fabs(output[i][j][k] - expected_output[i][j][k]) < tolerance);
            }
        }
    }
    printf("layer norm test passed\n");
}

void test_custom_transpose() {
    float input[2][3][5] = {
        {{1, 2, 3, 4, 5}, {6, 7, 8, 9, 10}, {11, 12, 13, 14, 15}},
        {{16, 17, 18, 19, 20}, {21, 22, 23, 24, 25}, {26, 27, 28, 29, 30}}
    };
    float output[3][2][5];
    float expected_output[3][2][5] = {
        {{1, 2, 3, 4, 5}, {16, 17, 18, 19, 20}},
        {{6, 7, 8, 9, 10}, {21, 22, 23, 24, 25}},
        {{11, 12, 13, 14, 15}, {26, 27, 28, 29, 30}}
    };
    
    custom_transpose(input, output, 0, 1, 5);
    
    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 2; ++j) {
            for(int k = 0; k < 5; ++k) {
                assert(output[i][j][k] == expected_output[i][j][k]);
            }
        }
    }

    printf("transpose test passed\n");
}

void custom_permute(float32_t* input, float32_t* output, int* dims, int size1, int size2, int size3) {
    int new_size1 = size1, new_size2 = size2, new_size3 = size3;

    // Adjust the sizes based on the new dimensions
    if (dims[0] == 0) new_size1 = size1;
    else if (dims[0] == 1) new_size1 = size2;
    else new_size1 = size3;

    if (dims[1] == 0) new_size2 = size1;
    else if (dims[1] == 1) new_size2 = size2;
    else new_size2 = size3;

    if (dims[2] == 0) new_size3 = size1;
    else if (dims[2] == 1) new_size3 = size2;
    else new_size3 = size3;

    for (int i = 0; i < new_size1; ++i) {
        for (int j = 0; j < new_size2; ++j) {
            for (int k = 0; k < new_size3; ++k) {
                int old_i = (dims[0] == 0) ? i : ((dims[0] == 1) ? j : k);
                int old_j = (dims[1] == 0) ? i : ((dims[1] == 1) ? j : k);
                int old_k = (dims[2] == 0) ? i : ((dims[2] == 1) ? j : k);
                output[i*new_size2*new_size3 + j*new_size3 + k] = input[old_i*size2*size3 + old_j*size3 + old_k];
            }
        }
    }
}

void test_custom_permute() {
    float32_t input[2*3*5] = {
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29
    };
    float32_t expected_output[2*5*3] = {
        0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14,
        15, 20, 25, 16, 21, 26, 17, 22, 27, 18, 23, 28, 19, 24, 29
    };
    float32_t output[2*5*3];
    int dims[3] = {0, 2, 1};

    custom_permute(input, output, dims, 2, 3, 5);

    // assert to verify the result
    for (int i = 0; i < 2*5*3; ++i) {
        assert(output[i] == expected_output[i]);
    }

    printf("permute passed.\n");
}

void test_custom_permute2() {
    float32_t input[2*3*5] = {
        0, 1, 2, 3, 4,
        5, 6, 7, 8, 9,
        10, 11, 12, 13, 14,
        15, 16, 17, 18, 19,
        20, 21, 22, 23, 24,
        25, 26, 27, 28, 29
    };
    float32_t expected_output[3*2*5] = {
        0.,  1.,  2.,  3.,  4.,
        15., 16., 17., 18., 19.,
        5.,  6.,  7.,  8.,  9.,
        20., 21., 22., 23., 24.,
        10., 11., 12., 13., 14.,
        25., 26., 27., 28., 29.,
    };
    float32_t output[3*2*5];
    int dims[3] = {1, 0, 2};

    custom_permute(input, output, dims, 2, 3, 5);

    // assert to verify the result
    for (int i = 0; i < 3*2*5; ++i) {
        assert(output[i] == expected_output[i]);
    }

    printf("permute2 passed.\n");
}

void custom_bmm(const float32_t *input, const float32_t *input2, float32_t *output, const int batch_size, const int m, const int n, const int k) {
    for (int i = 0; i < batch_size; ++i) {
        arm_matrix_instance_f32 mat1;
        arm_matrix_instance_f32 mat2;
        arm_matrix_instance_f32 mat_out;
        
        arm_mat_init_f32(&mat1, m, k, (float32_t *)(input + i * m * k));
        arm_mat_init_f32(&mat2, k, n, (float32_t *)(input2 + i * k * n));
        arm_mat_init_f32(&mat_out, m, n, output + i * m * n);
        
        arm_mat_mult_f32(&mat1, &mat2, &mat_out);
    }
}

void test_custom_bmm() {
    const int batch_size = 2;
    const int m = 3;
    const int n = 4;
    const int k = 5;

    float32_t input[] = { // batch_size * m * k
        // Batch 1
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        // Batch 2
        16, 17, 18, 19, 20,
        21, 22, 23, 24, 25,
        26, 27, 28, 29, 30,
    };

    float32_t input2[] = { // batch_size * k * n
        // Batch 1
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
        17, 18, 19, 20,
        // Batch 2
        21, 22, 23, 24,
        25, 26, 27, 28,
        29, 30, 31, 32,
        33, 34, 35, 36,
        37, 38, 39, 40,
    };

    float32_t output[batch_size * m * n];
    float32_t expected[] = { // batch_size * m * n
        175.0f, 190.0f, 205.0f, 220.0f,
        400.0f, 440.0f, 480.0f, 520.0f,
        625.0f, 690.0f, 755.0f, 820.0f,
        2650.0f, 2740.0f, 2830.0f, 2920.0f,
        3375.0f, 3490.0f, 3605.0f, 3720.0f,
        4100.0f, 4240.0f, 4380.0f, 4520.0f
    };

    custom_bmm(input, input2, output, batch_size, m, n, k);

    for (int i = 0; i < batch_size * m * n; ++i) {
        assert(fabs(output[i] - expected[i]) < 1e-6);
    }

    printf("bmm passed.\n");
}


void custom_chunk(float input[2][3][5], int chunks, int dim, float output[3][2][3][2]) {
    int chunk_sizes[3] = {2, 2, 1};  // Sizes of each chunk along the last dimension
    int i, j, k, l;
    
    // Split the input tensor into chunks
    for (i = 0; i < 2; i++) {
        for (j = 0; j < 3; j++) {
            for (k = 0; k < 5; k++) {
                int chunk_idx = k / 2;
                if (chunk_idx >= 2) chunk_idx = 2;  // Adjust index for the last chunk
                int chunk_position = k % 2;
                if (chunk_position < chunk_sizes[chunk_idx]) {
                    output[chunk_idx][i][j][chunk_position] = input[i][j][k];
                }
            }
        }
    }
}

void test_custom_chunk() {
    float input[2][3][5] = {
        {
            {0, 1, 2, 3, 4},
            {5, 6, 7, 8, 9},
            {10, 11, 12, 13, 14}
        },
        {
            {15, 16, 17, 18, 19},
            {20, 21, 22, 23, 24},
            {25, 26, 27, 28, 29}
        }
    };
    
    // Allocate space for output tensor chunks
    float output[3][2][3][2] = {0};  // Initialize all values to 0

    // Call custom_chunk function
    custom_chunk(input, 3, -1, output);

    // Expected output based on the PyTorch results
    float expected_output[3][2][3][2] = {
        {
            {{0, 1}, {5, 6}, {10, 11}},
            {{15, 16}, {20, 21}, {25, 26}}
        },
        {
            {{2, 3}, {7, 8}, {12, 13}},
            {{17, 18}, {22, 23}, {27, 28}}
        },
        {
            {{4, 0}, {9, 0}, {14, 0}},  // Note: The last dimension size is 1 for the last chunk, so the second value is unspecified
            {{19, 0}, {24, 0}, {29, 0}}
        }
    };

    // Assert that the custom_chunk output matches the expected output
    for (int chunk_idx = 0; chunk_idx < 3; chunk_idx++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 2; k++) {
                    assert(output[chunk_idx][i][j][k] == expected_output[chunk_idx][i][j][k]);
                }
            }
        }
    }

    printf("chunk passed.\n");
}

int main() {
    test_custom_conv1d();
    test_custom_liner();
    test_custom_layerNorm();
    test_custom_transpose();
    test_custom_permute();
    test_custom_permute2();
    test_custom_bmm();
    test_custom_chunk();
    return 0;
}
