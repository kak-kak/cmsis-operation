#include "arm_math.h"  // Include CMSIS-DSP library

#define INPUT_FEATURES 16
#define OUTPUT_FEATURES 64

void custom_linear(
    float32_t* input,
    float32_t* weights,
    float32_t* bias,
    float32_t* output
) {
    // Create matrix instances for CMSIS-DSP
    arm_matrix_instance_f32 mat_input, mat_weights, mat_output;

    // Initialize matrix instances
    arm_mat_init_f32(&mat_input, 1, INPUT_FEATURES, input);
    arm_mat_init_f32(&mat_weights, INPUT_FEATURES, OUTPUT_FEATURES, weights);
    arm_mat_init_f32(&mat_output, 1, OUTPUT_FEATURES, output);

    // Perform matrix multiplication
    arm_mat_mult_f32(&mat_input, &mat_weights, &mat_output);

    // Add bias
    for (uint32_t i = 0; i < OUTPUT_FEATURES; ++i) {
        output[i] += bias[i];
    }
}

int main() {
    float32_t input[INPUT_FEATURES];  // Initialize with your input data
    float32_t weights[INPUT_FEATURES * OUTPUT_FEATURES];  // Initialize with your weight data
    float32_t bias[OUTPUT_FEATURES];  // Initialize with your bias data
    float32_t output[OUTPUT_FEATURES];

    custom_linear(input, weights, bias, output);

    // output now holds the result of the linear transformation
    return 0;
}
