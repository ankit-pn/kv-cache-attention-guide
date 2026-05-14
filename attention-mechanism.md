# The Attention Mechanism: Q, K, V Explained

## What Are Q, K, V?

Every token generates three vectors from its embedding:

| Vector | Role | Analogy |
|---|---|---|
| **Q** (Query) | "What am I looking for?" | A search query |
| **K** (Key) | "What information do I contain?" | A document title / index |
| **V** (Value) | "What is my actual content?" | The document body itself |

## The Core Computation

```python
Q = X @ W_Q    # [N, d_k] — query projection
K = X @ W_K    # [N, d_k] — key projection
V = X @ W_V    # [N, d_k] — value projection

# For each token, compute relevance score against all previous tokens
scores = Q @ K.T              # [N, N] — similarity matrix
scores /= sqrt(d_k)           # Scale (prevent softmax saturation)
scores += causal_mask         # Apply causal mask
weights = softmax(scores)     # [N, N] — each row sums to 1
out = weights @ V             # [N, d_k] — weighted blend
```

Each row of `weights` tells a token: "here's how much to blend each previous token's information into your representation."

## Why We Need All Three

- **K and V** represent the **past** — they're computed once and stored forever
- **Q** represents the **present moment** — the current token's intent
- Without Q, every token would attend to the past identically (useless)
- Without K and V, there's no past to attend to (impossible)

## Q Is One-Time Use; K and V Are Reusable

At position N:
- **Qₙ** is computed from token N, used for token N's attention, then **discarded forever**
- **Kₙ** and **Vₙ** are computed from token N, but **every future token (N+1, N+2, ...) needs to read them**

This is why only K and V are cached — they have "many readers, one writer."
