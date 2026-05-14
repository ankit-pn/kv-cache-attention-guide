# My Confusions and What I Learned

This document captures the questions, doubts, and confusions I had during our conversation and the solutions/insights that resolved them.

---

## Confusion 1: Why Do We Only Cache K and V, Not Q?

**My question:** If the KV cache stores history, and we need it to produce new tokens, why don't we need Q from past tokens?

**Answer:** Because Q is one-time use per token. Token N computes Q_N once, attends to the past, and gets its hidden state. From that point on, Q_N is never needed again. But K_N and V_N will be queried by every future token (N+1, N+2, ...) — so they must persist.

## Confusion 2: In Multi-Head Attention, Do We Make 3h Copies of the Vector?

**My question:** If we have h heads, do we make 3h copies of the vector multiplied by 3h weight matrices?

**Answer:** No. We have **3 fused matrices** (W_Q, W_K, W_V) each of shape [d, d]. One big matmul per projection, then split into h heads by reshaping. Each head sees a **different d/h-dimensional slice** of the same token — it's a split, not a copy.

## Confusion 3: Is the Prefill Self-Attention Diagonal the Same as Decode?

**My question:** At layer 1, don't prefill and decode differ because prefill's position N attends to itself but decode's new token doesn't have itself in the cache yet?

**Answer:** Yes, I was right. This is a real mathematical discrepancy:
- Prefill: softmax over N+1 entries (including self)
- Decode: softmax over N entries (no self)

This difference is **not widely documented** in the literature but is empirically negligible because:
1. The self-attention diagonal weight is ~0.02 (model learns it doesn't help prediction)
2. After layer 1, KV is appended and convergence happens quickly
3. Production systems (vLLM, TensorRT-LLM) rely on this working correctly

## Confusion 4: Why Compute Q for All Tokens During Prefill, Not Just the Last One?

**My question:** If only the last token's hidden state feeds the LM Head, why compute Q for earlier tokens?

**Answer:** Because in multi-layer models, token 0's Q at layer 0 determines its hidden state at layer 0, which becomes token 0's K and V at layer 1. The last token at layer 1 attends to token 0's K and V. If token 0's Q was wrong, its K and V at layer 1 would be wrong, and the last token's attention would be corrupted.

**1-layer counterfactual:** In a 1-layer model, you genuinely only need Q for the last token. But real models have 43-61 layers.

## Confusion 5: Can We Produce a New Token With Only the KV Cache, Without Q?

**My question:** If we have all the KV cache, can we skip computing Q entirely?

**Answer:** Not in general. The KV cache is passive — it stores past information but needs someone to query it. That query is Q, which comes from a token. Even a 100% cache hit still requires running the last prompt token through all layers (providing Q at each layer) to produce the final hidden state.

**Theoretical exception:** If we cached the final hidden state of the last token at layer 60, we could skip Q entirely for exact matches. But that's ~0.5 MB per sequence and rarely worth the gain.

## Confusion 6: What Happens When Two Prompts Diverge and Then Converge Again?

**My question:** If prompts A and B match, then differ at position m, then have identical tokens from m+2 onwards — can we reuse KV from m+2?

**Answer:** No. Once a different token appears, its hidden state at every layer will differ from the original. Even though the input tokens are the same from m+2, their attention context includes the divergent position m. The hidden state divergence cascades through all layers. Only positions [0..m-1] can share KV cache.

## Confusion 7: During Decode, Do We Pass 1 Token or the Whole Context?

**My question:** When decoding, do we feed just the new token or the entire sequence + new token to the model?

**Answer:** Just **1 new token** enters the model. The entire past enters through the KV cache. The model's Q is computed from just the new token, then it scores that Q against all cached K of previous tokens. This is the entire purpose of the KV cache — avoiding re-feeding the full context.

## Confusion 8: Prefix Extension vs Re-prefill

**My question:** When a user adds 10K new tokens to a 100K conversation, do we decode 10K times or prefill?

**Answer:** We **prefill** the new 10K tokens in parallel (one big forward pass for all of them using the cached K,V of the existing 100K). Then we **decode** the assistant's response one token at a time. 10,000 sequential decode steps would be much slower than one parallel prefill.
