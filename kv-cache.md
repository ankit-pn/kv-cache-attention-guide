# The KV Cache — What Gets Cached and Why

## What We Cache

For each position at each layer:

```
Layer 0:  K₀["The","cat","sat","on"]  → stored
          V₀["The","cat","sat","on"]  → stored
Layer 1:  K₁["The","cat","sat","on"]  → stored
          V₁["The","cat","sat","on"]  → stored
...
```

Each layer has its **own independent** KV cache because:
1. Each layer has **different weight matrices** (W_K⁰ ≠ W_K¹)
2. Each layer's input is **different** (the previous layer's output)

## What We DON'T Cache

- **Q** — one-time use per token
- **Hidden states** — only needed to compute the next layer's K,V during the original pass

## Why Only K and V?

| Vector | Needed at time N? | Needed at time N+1? | Needed at time N+2? | Cache it? |
|---|---|---|---|---|
| Qₙ | ✅ for scoring | ❌ never | ❌ never | No |
| Kₙ | ✅ as scoring target | ✅ for future Q's | ✅ for future Q's | **Yes** |
| Vₙ | ✅ in value blend | ✅ in value blend | ✅ in value blend | **Yes** |

**K and V are read many times by future tokens. Q is read once and never needed again.**

## The Layer-by-Layer KV Cache

Each layer of a 61-layer model has its own KV cache:

```
KV Cache for ONE token at position N:
┌────────────────────────────────────────────────────────────┐
│ Layer 0:  K₀ = embed(x) @ W_K⁰   →  stored                │
│           V₀ = embed(x) @ W_V⁰   →  stored                │
├────────────────────────────────────────────────────────────┤
│ Layer 1:  K₁ = hidden_0(x) @ W_K¹   →  stored              │
│           V₁ = hidden_0(x) @ W_V¹   →  stored              │
├────────────────────────────────────────────────────────────┤
│ ...                                                         │
├────────────────────────────────────────────────────────────┤
│ Layer 60: K₆₀ = hidden_59(x) @ W_K⁶⁰ →  stored            │
│           V₆₀ = hidden_59(x) @ W_V⁶⁰ →  stored            │
└────────────────────────────────────────────────────────────┘
```

When a new token is decoded, it computes Q at every layer and queries each layer's cache independently.

## The Three-Tier Cache Hierarchy (Production)

| Tier | Media | Bandwidth | Use case |
|---|---|---|---|
| GPU HBM | GPU memory | ~3.35 TB/s | Active requests |
| CPU RAM | DRAM | ~50 GB/s | Warm cache (recent sessions) |
| NVMe Disk | SSD | ~10 GB/s | Cold cache (popular shared prefixes) |

The on-disk cache exists not because KV doesn't fit in HBM, but to **avoid re-prefilling** shared prefixes across sessions.
