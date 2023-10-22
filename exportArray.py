import numpy as np

depth = 3
ch = 4
width = 5

# Create a 3D NumPy array using np.arange
python_matrix = np.arange(depth * ch * width, dtype=np.float32).reshape(depth , ch , width)

# Save to binary file
python_matrix.tofile('python_matrix.bin')

# Verification by printing elements (for debugging purposes)
for i in range(python_matrix.shape[0]):
    for j in range(python_matrix.shape[1]):
        print(" ".join(str(python_matrix[i, j, k]) for k in range(python_matrix.shape[2])))
    print()
