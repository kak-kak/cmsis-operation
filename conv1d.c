#include "arm_math.h"  // Include CMSIS-DSP library

#define INPUT_CHANNELS 16
#define OUTPUT_CHANNELS 64
#define KERNEL_SIZE 11
#define INPUT_SIZE 250
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)  // Assuming no padding and stride of 1

void custom_conv1d(
    float32_t* input, 
    float32_t* kernel, 
    float32_t* output, 
    uint32_t input_channels, 
    uint32_t output_channels,
    uint32_t input_size,
    uint32_t kernel_size
) {
    for (uint32_t oc = 0; oc < output_channels; ++oc) {
        for (uint32_t ic = 0; ic < input_channels; ++ic) {
            float32_t single_output[OUTPUT_SIZE];
            arm_conv_f32(&input[ic * input_size], input_size, &kernel[oc * input_channels * kernel_size + ic * kernel_size], kernel_size, single_output);
            for (uint32_t i = 0; i < OUTPUT_SIZE; ++i) {
                output[oc * OUTPUT_SIZE + i] += single_output[i];  // Accumulate to the final output
            }
        }
    }
}

int main() {
    float32_t input[INPUT_CHANNELS * INPUT_SIZE];  // Initialize with your input data
    float32_t kernel1[INPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL_SIZE];  // Initialize with your kernel weights for conv1
    float32_t kernel2[OUTPUT_CHANNELS * OUTPUT_CHANNELS * KERNEL_SIZE];  // Initialize with your kernel weights for conv2
    float32_t output1[OUTPUT_CHANNELS * OUTPUT_SIZE];
    float32_t output2[OUTPUT_CHANNELS * OUTPUT_SIZE];

    // Zero-initialize the output buffers
    memset(output1, 0, sizeof(output1));
    memset(output2, 0, sizeof(output2));

    custom_conv1d(input, kernel1, output1, INPUT_CHANNELS, OUTPUT_CHANNELS, INPUT_SIZE, KERNEL_SIZE);
    custom_conv1d(output1, kernel2, output2, OUTPUT_CHANNELS, OUTPUT_CHANNELS, OUTPUT_SIZE, KERNEL_SIZE);

    // output2 now holds the result of passing input through conv1 and conv2
    return 0;
}
