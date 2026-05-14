# How an Autoregressive Transformer Generates Tokens

## The Two Phases

### Phase 1: Prefill
All prompt tokens are processed in **parallel** in one big forward pass.

```
Input: "The capital of France is"  (5 tokens)

Layer 1:
  Q = embed @ W_Q    → one big matmul over ALL 5 tokens
  K = embed @ W_K    → one big matmul
  V = embed @ W_V    → one big matmul

  scores = Q @ K.T   → [5×5] causal masked
  weights = softmax(scores)
  out = weights @ V

  K,V are stored in KV cache for future decode steps

Layer 2..61: same process using hidden states from previous layer

Final layer → LM Head → logits → sample → "Paris" (first generated token)
```

### Phase 2: Decode
Tokens are generated **one at a time**. Each new token attends to ALL previous tokens using the cached K,V.

```
KV cache already has 5 entries from prefill.

Step 1: embed("Paris") → [1, d]
Step 2: q = x @ W_Q → attend to K_cache of 5 tokens
Step 3: Append K_new, V_new to cache (now 6 entries)
Step 4: LM Head → sample → "is"

Step 5: embed("is") → [1, d]
Step 6: q = x @ W_Q → attend to K_cache of 6 tokens
Step 7: Append → cache now has 7 entries
... continues until <EOS> or max tokens
```

## Key Mental Model

**The KV cache is the model's notepad.** Instead of re-reading the entire conversation history every time it needs to produce the next word, it keeps its reading notes (K and V) and consults them directly. The notes grow with each word written, but the act of re-reading everything from scratch never happens.

## What Gets Cached

For **every token** at **every layer**, we store:
- **K** (Key): "What information do I contain?"
- **V** (Value): "What is my actual content?"

These two values per token per layer are all that future tokens need to attend to the past.

## The Initiator Problem

**You always need an existing token to produce the next one.** The causal chain is:

```
Tokenₙ → Q → Attention → Hidden → LM Head → Tokenₙ₊₁ → Q → Attention → ...
```

Even with 100% KV cache hit, you must run **1 decode step** for the last prompt token through all layers to produce the first generated token. Because hidden states at layer 60 are NOT cached — only K and V are.
