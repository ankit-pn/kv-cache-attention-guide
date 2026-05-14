# Causal Masking — Why Old Tokens Stay Frozen

## The Causal Mask

```python
causal_mask = [[  0.,  -inf, -inf, -inf, -inf],
               [  0.,   0.,  -inf, -inf, -inf],
               [  0.,   0.,   0.,  -inf, -inf],
               [  0.,   0.,   0.,   0.,  -inf],
               [  0.,   0.,   0.,   0.,   0.]]
```

Token at position `i` can only see positions `0..i`. Future positions get `-inf`, which becomes 0 in softmax. This is how autoregressive prediction works — token `i` predicts token `i+1` using only positions up to `i`.

## Causal Mask Makes Prefix Caching Possible

Because old tokens can't see new tokens, their representations are **permanently frozen** after their initial computation:

```
Position:    0      1      2      3  (OLD)    4      5  (NEW)
Token:      "The"  "cat"  "sat"  "on"        "mat"  "explain"

Causal Mask:
          ┌──────────────────────────────────────────────┐
          │  █  │  -∞  │  -∞  │  -∞  │  -∞  │  -∞  │  Token 0
          │  █  │   █  │  -∞  │  -∞  │  -∞  │  -∞  │  Token 1
          │  █  │   █  │   █  │  -∞  │  -∞  │  -∞  │  Token 2
OLD ──►  │  █  │   █  │   █  │   █  │  -∞  │  -∞  │  Token 3
          │  █  │   █  │   █  │   █  │   █  │  -∞  │  Token 4 (last OLD)
NEW ──►  │  █  │   █  │   █  │   █  │   █  │   █  │  Token 5 "explain"
          └──────────────────────────────────────────────┘
```

**Token 3 ("on") cannot see Token 4 or Token 5.** Its representation was finalized during prefill. Adding new tokens doesn't change it.

## What This Enables

| Feature | Enabled by causal masking |
|---|---|
| **KV cache** | Old tokens don't need to recompute when new tokens arrive |
| **Prefix caching** | Hash a prefix, share its KV cache across requests |
| **On-disk caching** | Store frozen K,V on disk; load ~0.3s later |
| **Parallel prefill** | All positions computed in one batch with causal mask |

Without the causal mask, every new token would require recomputing ALL previous tokens' representations — making KV cache impossible.
