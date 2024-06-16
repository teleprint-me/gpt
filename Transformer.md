The following are the neural network building blocks (functions with learnable parameters) from which transformers are made. Full architectures featuring these building blocks are presented in the next section. (By a slight abuse of notation, we identify 𝑉 with the set {1, 2, . . . , 𝑁V }.)

Token embedding. The token embedding learns to represent each vocabulary element as a vector in ℝ^𝑑_e; see Algorithm 1.

| Algorithm 1: Token embedding. |
| ----------------------------- |
| Input: 𝑣 ∈ 𝑉 ≅ [𝑁_𝑉], a token ID. |
| Output: 𝒆 ∈ ℝ^𝑑_e, the vector representation of the Token. |
| Parameters: 𝑾_𝒆 ∈ ℝ^(𝑑_e × 𝑁_𝑉), the token embedding matrix. |
| 1 return 𝒆 = 𝑾_𝒆[:,𝑣] |

| Symbol | Type | Explanation |
| ------ | ---- | ----------- |
| 𝑉 | ≅ [ 𝑁_V ] | vocabulary |
| 𝑾_𝒆 | ∈ ℝ^(𝑑_e × 𝑁_V) | token embedding matrix |

 Algorithm 2: Positional embedding.
  Input:  ∈ [ max ], position of a token in
         the sequence.
  Output: 𝒆 𝒑 ∈ ℝ𝑑e , the vector
            representation of the position.
  Parameters: 𝑾𝒑 ∈ ℝ𝑑e ×max , the positional
                embedding matrix.
1 return 𝒆 𝒑 = 𝑾 𝒑 [:, ]