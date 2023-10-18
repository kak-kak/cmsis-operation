import numpy as np

def custom_conv1d(input, weight, bias=None):
    # Get dimensions
    in_channels, in_width = input.shape
    out_channels, _, kernel_width = weight.shape
    
    # Output width computation
    out_width = in_width - kernel_width + 1
    
    # Initialize output
    output = np.zeros((out_channels, out_width))
    
    # Compute convolution
    for o in range(out_channels):
        for w in range(out_width):
            for i in range(in_channels):
                output[o, w] += np.sum(input[i, w:w+kernel_width] * weight[o, i])
    
    # Add bias if provided
    if bias is not None:
        output += bias.reshape(-1, 1)
    
    return output

# Example usage:
input_data = np.array([[1, 2, 3, 4]])  # Shape: (1, 4)
weights = np.array([[[0.1, 0.2]]])    # Shape: (1, 1, 2)
bias = np.array([0.1])                # Shape: (1,)

output_data = custom_conv1d(input_data, weights, bias)
print(output_data)
