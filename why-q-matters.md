# Why Q Must Be Computed for Every New Token

## The Core Chain

```
Token → Q⁰ → Attention⁰ → Hidden⁰ → Q¹ → Attention¹ → Hidden¹ → ... → Q⁶⁰ → LM Head → next token
```

Q at layer L determines the hidden state at layer L. That hidden state becomes the **input** to layer L+1, which computes K_{L+1} and V_{L+1} from it. Without Q at every layer, the chain breaks.

## What Happens Without Q

At any layer L:

```python
# Without Q:
x = previous_layer_output
# We can't compute attention!
# We can't produce a hidden state!
# The next layer can't compute K or V from it!
# The LM Head never gets its input!
```

**Q is not optional.** It's the mechanism by which the new token defines its own identity at each layer and queries the past.

## 1-Layer Counterfactual

In a hypothetical 1-layer model, you only need Q for the last token:

```python
# 1-layer model:
K = embed @ W_K    # from embeddings (no attention needed)
V = embed @ W_V    # from embeddings (no attention needed)

# Only need Q for last token:
Q_last = embed[-1] @ W_Q
scores = Q_last @ K.T
hidden = scores @ V
next_token = LM_head(hidden)
```

But with 61 layers, K at layer 1 depends on hidden states from layer 0, which requires Q at layer 0 for every position. So you need Q for EVERY token at EVERY layer.

## The One Exception

If you cached the **final hidden state** of the last token (layer 60 output), you could theoretically skip Q entirely for an exact 100% prefix match:

```python
# If hidden_60 is cached:
logits = hidden_60["last_prompt_token"] @ W_lm_head
next_token = sample(logits)   # zero compute!
```

This is possible but **not done in practice** because:
1. ~0.5 MB of extra storage per cached prefix
2. Most cache hits are partial, not 100%
3. One decode step (~50 μs) is already negligible
