#include <stdio.h>
#include <stdlib.h>

// Global variables
float* token_embedding;
int    vocabulary_size     = 32000;
int    embedding_dimension = 768;

void init(int size, int dimension) {
    // Initialize token embedding matrix
    token_embedding = (float*) malloc((vocabulary_size + 1) * dimension);

    if (!token_embedding) {
        printf("Error: Unable to allocate memory for the token embedding.\n");
        exit(EXIT_FAILURE);
    }

    vocabulary_size     = size;
    embedding_dimension = dimension;
}

void set_parameter(int index, float value) {
    if (index < 0 || index >= (vocabulary_size + 1)) {
        printf("Error: Index out of range.\n");
        exit(EXIT_FAILURE);
    }

    token_embedding[index * embedding_dimension] = value;
}

float* get_token_vector(int index) {
    if (index < 0 || index >= vocabulary_size) {
        printf("Error: Index out of range.\n");
        exit(EXIT_FAILURE);
    }

    return &token_embedding[index * embedding_dimension];
}

int main() {
    // Initialize the token embedding matrix
    init(1000, 512);

    // Set some example parameters
    for (int i = 0; i < vocabulary_size + 1; ++i) {
        set_parameter(i, rand() % 10 - 5.f);
    }

    float* token_vector = get_token_vector(42);

    printf("Token ID: 42\n");
    for (int i = 0; i < embedding_dimension; ++i) {
        printf("Component %d: %.3f\n", i, token_vector[i]);
    }

    return EXIT_SUCCESS;
}
