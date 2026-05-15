# KV Cache & Attention — A Deep Dive

Everything I learned (and questioned) about how autoregressive transformers work under the hood.

## Table of Contents

1. [How an Autoregressive Transformer Generates Tokens](./autoregressive-transformer.md)
2. [The Attention Mechanism: Q, K, V Explained](./attention-mechanism.md)
3. [The KV Cache — What Gets Cached and Why](./kv-cache.md)
4. [How to read the attention computation — step by step](./attention-algebra.md)
5. [Multi-Head Attention — Split, Not Copy](./multi-head-attention.md)
6. [Causal Masking — Why Old Tokens Stay Frozen](./causal-masking.md)
7. [Mixture of Experts (MoE) — Explained](./mixture-of-experts.md)
8. [Prefix Caching — What Gets Reused and What Doesn't](./prefix-caching.md)
9. [Why Q Must Be Computed for Every New Token](./why-q-matters.md)
10. [My Confusions and What I Learned](./confusions.md)
11. [Glossary](./glossary.md)
