# Prefix Caching — What Gets Reused and What Doesn't

## Scenario: Two Prompts Sharing a Prefix

```
Prompt A: The cat sat on the mat
Prompt B: The cat ran on the mat
          ───── match ──┘
```

**Only positions 0..1 ("The cat") can share KV cache.**

After divergence (position 2: "sat" vs "ran"):

- Position 2 has different K,V because its input token is different
- Position 3 "on" has the SAME input token, but its **attention context** now includes the divergent position 2. Since its attention scores depend on ALL previous tokens (including the different one at position 2), its hidden state differs → different K,V at the next layer
- Position 4 "the", position 5 "mat" — same story. The divergence cascades
- Every layer compounds the difference

## Key Insight

> **Once diverged, always diverged.** Even if the tokens become identical again, the hidden states will differ because attention context includes the divergent positions. The causal mask prevents old tokens from seeing new ones, but it doesn't prevent the effect of a DIFFERENT token in the past.

## What Happens When We Match

When the server finds a matching prefix:

```python
# Only for the NEW (non-matching) tokens:
Q_new = X_new @ W_Q    # compute Q for new tokens
K_new = X_new @ W_K    # compute K for new tokens
V_new = X_new @ W_V    # compute K for new tokens

# Load pre-computed K,V for matching prefix
K_all = concat(K_prefix_cached, K_new)
V_all = concat(V_prefix_cached, V_new)

# New tokens attend to ALL (cached + new)
scores = Q_new @ K_all.T
```

**The old tokens' Q was already computed during the original request. It's never recomputed.** Their K,V are frozen and ready to serve.

## What's NOT Cached (And Why We Still Need 1 Step)

Even with 100% prefix match, we don't cache the final hidden state of the last prompt token. So we must run:

```
1 decode step: last prompt token through 61 layers → LM Head → next token

This is the irreducible minimum — ~50 μs on an H100.
```
