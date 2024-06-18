# Formal Algorithms for Transformers

Anyone who says this is easy is inexperienced, has experience and lacks the self awareness of the amount of effort it took to acquire their skills, and/or has an eidetic memory. In any case, I can't take what they say with any form of seriousness and must accept their words as a grain of salt.

## Algorithm 1: Token Embedding

Algorithm 1 describes a simple lookup operation that converts each unique vocabulary element into
its respective vector representation in the embeddings space using a learnable parameter, the token
embedding matrix `𝑾_𝒆`.

| Algorithm 1. | Token embedding.        | Description                             |
| ------------ | ----------------------- | --------------------------------------- |
| Input        | `𝑣` ∈ `𝑉` ≅ `[𝑁_𝑉]`     | a token ID.                             |
| Output       | `𝒆` ∈ `ℝ^𝑑_e`           | the vector representation of the Token. |
| Parameters   | `𝑾_𝒆` ∈ `ℝ^(𝑑_e × 𝑁_𝑉)` | the token embedding matrix.             |
| 1 return     | `𝒆 = 𝑾_𝒆[:,𝑣]`          |                                         |

### Symbol Explanation

| Symbol | Type | Explanation |
| --- | --- | --- |
| `𝑉` | ≅ `[𝑁_V]` | vocabulary |
| `𝑁_V` | ∈ `ℕ` | vocabulary size |
| `𝒆` | ∈ `ℝ^𝑑_e` | vector representation / embedding of a token |
| `𝑑_e` | Embedding Dimensionality | The dimensionality of the embeddings space. |
| `𝑾_𝒆` | ∈ `ℝ^(𝑑_e × 𝑁_V)` | token embedding matrix is a learnable parameter |
| `𝑉∗` | = `⋃◌᪲_l=0 (𝑉^l)` | set of token sequences; elements include e.g. sentences or documents |

In this section, we have outlined the token embedding algorithm and provided a brief explanation for
each symbol used in the algorithm. The next step would be to implement this algorithm in code, such
as Python or C, and document any additional insights gained during implementation.

## Token Embedding C Implementation

```c
#include <stdio.h>

// Global variables
float *token_embedding;
int vocabulary_size=32000;
int embedding_dimension;

void init(int size, int dimension) {
    // Initialize token embedding matrix
    token_embedding = (float*)malloc((vocabulary_size + 1) * dimension);

    if (!token_embedding) {
        printf("Error: Unable to allocate memory for the token embedding.\n");
        exit(EXIT_FAILURE);
    }

    vocabulary_size = size;
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

    float *token_vector = get_token_vector(42);

    printf("Token ID: 42\n");
    for (int i = 0; i < embedding_dimension; ++i) {
        printf("Component %d: %.3f\n", i, token_vector[i]);
    }

    return EXIT_SUCCESS;
}
```

## Algorithm 2: Positional embedding.
Input:  ∈ [ max ], position of a token in
        the sequence.
Output: 𝒆 𝒑 ∈ ℝ𝑑e , the vector
        representation of the position.
Parameters: 𝑾𝒑 ∈ ℝ𝑑e ×max , the positional
            embedding matrix.
1 return 𝒆 𝒑 = 𝑾 𝒑 [:, ]
