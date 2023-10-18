#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "arm_math.h"  // Include CMSIS-DSP library

// ... (Insert custom functions here)

// Helper function to compare floating point arrays with some tolerance
int array_equals(float32_t* a, float32_t* b, uint32_t size, float32_t tolerance) {
    for (uint32_t i = 0; i < size; ++i) {
        if (fabs(a[i] - b[i]) > tolerance) {
            return 0;
        }
    }
    return 1;
}

int main() {
    // Test custom_conv1d
    {
        float32_t input[] = {1, 2, 3, 4};
        float32_t kernel[] = {0.1, 0.2};
        float32_t output[5] = {0};
        float32_t expected_output[] = {0.1, 0.4, 0.7, 1.0, 0.8};
        custom_conv1d(input, kernel, output, 1, 2, 4, 2);
        assert(array_equals(output, expected_output, 5, 0.001));
    }

    // Test custom_linear
    {
        float32_t input[] = {1, 2, 3, 4};
        float32_t weights[] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8};
        float32_t bias[] = {0.1, 0.2};
        float32_t output[2] = {0};
        float32_t expected_output[] = {3.0, 3.4};
        custom_linear(input, weights, bias, output);
        assert(array_equals(output, expected_output, 2, 0.001));
    }

    // Test custom_layer_norm
    {
        float32_t input[] = {1, 2, 3, 4};
        float32_t weight[] = {1, 1, 1, 1};
        float32_t bias[] = {0, 0, 0, 0};
        float32_t output[4] = {0};
        float32_t expected_output[] = {-1.3416, -0.4472, 0.4472, 1.3416};
        custom_layer_norm(input, weight, bias, output);
        assert(array_equals(output, expected_output, 4, 0.001));
    }

    printf("All tests passed!\n");

    return 0;
}
