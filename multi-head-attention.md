# Multi-Head Attention — Split, Not Copy

## Common Misconception: "We make h copies of the vector"

**Not true.** One fused projection, then split:

```python
# What people imagine (WRONG):
Q_0 = X @ W_Q_0    # separate matmul per head
Q_1 = X @ W_Q_1
...
Q_h = X @ W_Q_h    # h separate matmuls = slow

# What actually happens (RIGHT):
W_Q = concat(W_Q_0, W_Q_1, ..., W_Q_h)  # [d, d] — one fused matrix
Q = X @ W_Q                               # [N, d] — one matmul
Q = Q.reshape(N, h, d_k).transpose(0, 1)  # [h, N, d_k] — slice into heads
```

**We split the dimension budget, we don't copy.** A token of dimension d=4096 gets split into h=32 heads each of dimension d_k=128. Each head sees a **different 128-dimensional view** of the same token.

## The Analogy

Think of a team of specialists reading the same document:

| Head | Specialist | Reads |
|---|---|---|
| Head 0 | Grammar checker | Sentence structure |
| Head 1 | Fact checker | Entity relationships |
| Head 2 | Sentiment analyst | Emotional tone |
| Head 3 | Syntax analyzer | Subject-verb-object |
| ... | ... | ... |

Each one looks for different patterns in the same text. They don't make copies of the document — they just focus on different aspects.

## How Many Matrices?

For MHA with h=32, d=4096, d_k=128:

| Matrices | Shape | Total |
|---|---|---|
| W_Q | [d, d] | 1 fused matrix |
| W_K | [d, d] | 1 fused matrix |
| W_V | [d, d] | 1 fused matrix |
| W_O | [d, d] | 1 output mix |
| **Total** | | **4 matrices, 3 matmuls for projections** |

Not 3h matrices. Just 3 big ones.
