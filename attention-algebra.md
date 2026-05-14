# The Attention Algebra — Step by Step

## The Full Computation for One Layer During Prefill

```python
Input:  X ∈ R^(N × d)    (N = sequence length, d = hidden dim)

Step 1: Q = X @ W_Q      # [N, d] — every token asks "what am I looking for?"
Step 2: K = X @ W_K      # [N, d] — every token says "here's my key"
Step 3: V = X @ W_V      # [N, d] — every token says "here's my value"

Step 4: scores = Q @ K.T  # [N, N]
        # Cell (i,j) = how much should token i look at token j?

Step 5: scores /= sqrt(d)  # Scale to prevent softmax saturation

Step 6: scores += causal_mask  # Upper triangle becomes -inf
        # Token i can only see positions 0..i

Step 7: weights = softmax(scores)  # [N, N]
        # Each row sums to 1

Step 8: out = weights @ V  # [N, d]
        # Row i = weighted blend of values from tokens 0..i

Step 9: out = out @ W_O + X  # Output projection + residual
```

## Decode: Just 1 Token at a Time

```python
# KV cache already has N entries from prefill
x = embed(new_token)        # [1, d]

q = x @ W_Q                 # [1, d] — just 1 query
scores = q @ K_cache.T      # [1, N] — 1 vector × N matrix
weights = softmax(scores)   # [1, N]
out = weights @ V_cache     # [1, d]

# Append new K,V to cache
K_cache = concat(K_cache, x @ W_K)  # now N+1 entries
V_cache = concat(V_cache, x @ W_V)
```

The KV cache converts attention from O(N²) to O(N) per step.

## What the Matrix Shapes Mean

```
Prefill:      Q [N × d] @ K [d × N]  → scores [N × N]
              One big matrix multiply, very GPU-efficient.

Decode:       q [1 × d] @ K_cache [d × N]  → scores [1 × N]
              Vector × matrix. Much less compute, memory-bandwidth bound.
```

Prefill has high arithmetic intensity (N²) so it's **compute-bound**.
Decode is memory-bandwidth bound because it loads all cached K,V but only computes a vector multiply.
