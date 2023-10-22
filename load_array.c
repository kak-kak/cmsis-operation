#include <stdio.h>
#include <stdlib.h>

int main()
{
    size_t depth = 3;
    size_t ch = 4;
    size_t width = 5;

    FILE *fp = fopen("src_py/python_matrix.bin", "rb");
    if (fp == NULL)
    {
        printf("Failed to open file.\n");
        return 1;
    }

    float c_matrix[depth][ch][width];
    size_t read_elements = fread(c_matrix, sizeof(float), depth * ch * width, fp);

    if (read_elements != depth * ch * width)
    {
        printf("Failed to read the expected number of elements.\n");
        fclose(fp);
        return 1;
    }

    fclose(fp);

    // Verification by printing elements (for debugging purposes)
    for (int i = 0; i < depth; ++i)
    {
        for (int j = 0; j < ch; ++j)
        {
            for (int k = 0; k < width; ++k)
            {
                printf("%.1f ", c_matrix[i][j][k]);
            }
            printf("\n");
        }
        printf("\n");
    }

    return 0;
}
