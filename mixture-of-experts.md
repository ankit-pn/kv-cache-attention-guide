# Mixture of Experts (MoE) — Explained

## Where MoE Fits in a Transformer

A standard transformer layer has two parts:

```python
# Standard Dense Transformer Layer:
x = x + Attention(x)        # step 1: attention 
x = x + FFN(x)              # step 2: feed-forward ← THIS is replaced by MoE
```

The FFN is just two matrix multiplies with an activation:

```python
def ffn(x):
    h = gelu(x @ W_up)      # [d] → [4d]
    return h @ W_down        # [4d] → [d]
```

**MoE replaces this ONE FFN with MANY smaller FFNs + a router.**

---

## The Core Idea: Conditional Computation

| Model type | What happens | Compute cost | Knowledge capacity |
|---|---|---|---|
| **Dense (7B)** | Every token → 100% of FFN | High | Only 7B worth |
| **MoE (256 experts)** | Every token → 6 out of 256 experts | **Same** as dense | **671B worth** |

You get the knowledge capacity of a 671B model for the compute cost of a 13B model.

---

## The Router — What It Is and How It Works

The router is a single small linear layer:

```python
class Router(nn.Module):
    def __init__(self, input_dim=4096, num_experts=256):
        self.W = nn.Linear(input_dim, num_experts)   # [4096, 256]
    
    def forward(self, x):
        logits = self.W(x)        # [1, 256] — one score per expert
        return logits
```

~1M params, ~0.001% of the model. That's it.

## Walkthrough: The Token "bank"

```python
x = hidden_state_of("bank")        # [4096]

# Step 1: Router scores every expert
scores = router.W(x)               # [256]
# Expert 42:  5.9  ← HIGHEST
# Expert 203: 4.8
# Expert 3:   4.1
# Expert 0:   3.2
# Expert 255: 2.7
# Expert 18:  2.1
# Other 250:  < 2.0

# Step 2: Pick top-K (e.g., 6)
weights, indices = top_k(scores, 6)

# Step 3: Softmax normalize those 6 weights
weights = softmax([5.9, 4.8, 4.1, 3.2, 2.7, 2.1])
# Result:      [0.31, 0.25, 0.18, 0.13, 0.08, 0.05]
```

## What Each Expert Learned

Each expert is a tiny FFN with its OWN weight matrices:

```python
class Expert(nn.Module):
    def __init__(self, d_model=4096, d_expert=512):
        self.W_gate = nn.Linear(d_model, d_expert)
        self.W_down = nn.Linear(d_expert, d_model)
    
    def forward(self, x):
        return self.W_down(gelu(self.W_gate(x)))
```

After training, experts naturally specialize:

| Expert | Specializes in | Activated by tokens like |
|---|---|---|
| 42 | Finance | bank, money, loan, interest, credit |
| 203 | Geography | river, valley, mountain, coast, shore |
| 3 | Buildings/structures | office, building, branch, tower |
| 17 | Programming | def, import, class, function, return |
| 88 | Chemistry | atom, molecule, electron, bond |
| 155 | Animals | cat, dog, animal, species |

No one assigned these roles. **They emerged from training** because tokens about similar topics developed similar hidden states, and the router learned to route them to the same experts.

## The Weighted Blend

```python
output = 0.31 * expert_42(x)    # 31% financial perspective
       + 0.25 * expert_203(x)   # 25% geographical perspective
       + 0.18 * expert_3(x)     # 18% building perspective
       + 0.13 * expert_0(x)     # 13% general perspective
       + 0.08 * expert_255(x)   # 8% legal perspective
       + 0.05 * expert_18(x)    # 5% miscellaneous

output += shared_expert(x)      # Always-on expert for common knowledge

return x + output                # Residual connection
```

"bank" gets a 31% financial + 25% geographical + 18% building + ... interpretation. A perfect blend for the ambiguous word.

---

## DeepSeek's Innovations

### 1. Shared Expert (Always On)

```python
output = shared_expert(x) + blend_of_routed_experts(x)
```

- Captures **common knowledge** (grammar, syntax, basic patterns)
- Prevents all 256 routed experts from redundantly learning the same basics
- Proved irreplaceable — disabling it and activating one more routed expert **degrades performance**

### 2. Fine-Grained Experts

| | Standard MoE | DeepSeekMoE |
|---|---|---|
| Expert size | Large (d_ff = 16384) | Tiny (d_expert = 512) |
| Number of experts | 8-64 | 256-384 |
| Activated per token | 2 | 6 |
| Possible combinations | C(8,2) = 28 | C(256,6) = ~10¹² |

More combinations → more specialized routing → better performance.

### 3. Auxiliary-Loss-Free Load Balancing

**Problem:** The router can be lazy and route all tokens to the same expert.

**Old solution:** Add an auxiliary loss penalizing imbalance. But this conflicts with the main training objective and hurts performance.

**DeepSeek's solution:** No loss term. Instead, maintain a **bias** per expert and update it dynamically:

```python
# Before each batch:
for expert_i in range(num_experts):
    if load_i > ideal_load:    # Too many tokens → discourage
        bias_i -= 0.001
    elif load_i < ideal_load:  # Too few tokens → encourage
        bias_i += 0.001

# Use biased scores for routing:
scores = softmax(x @ W_router + bias)
```

No gradient interference, better performance, better balance.

---

## Comparison Across DeepSeek Models

| | V3/V3.2 | V4-Flash | V4-Pro |
|---|---|---|---|
| Total routed experts | 256 | 256 | 384 |
| Activated per token | 8 | 6 | 6 |
| Shared experts | 1 | 1 | 1 |
| Expert intermediate dim | — | 2048 | 3072 |
| Hidden dim | 7168 | 4096 | 7168 |

V4 dropped from 8→6 activated experts because each expert is more specialized and covers its domain better.

---

## How MoE Interacts with Attention

**They're completely independent.** MoE only replaces the FFN:

```
Layer:
  x = x + Attention(x)    ← Q/K/V, KV cache, causal mask — all unchanged
  x = x + MoE(x)          ← Router picks experts — entirely separate
```

MoE doesn't touch the KV cache. MoE doesn't affect attention. They operate on the same hidden state but are separate computations with separate weights.

## Serving Challenge

The model has 671B total params but only 37B active per token. The trick:

```
All 671B weights must be in GPU memory (across GPUs).
Only 37B are read from HBM per token.

GPUs must communicate: "I have expert 42, you have expert 17 —
  token needs both, send me its output."
```

This **expert parallelism** communication is the main inference overhead, solved by DeepSeek's MegaMoE kernel (1.5-1.96× speedup).

---

## TL;DR

```
Dense:    x → [Attention] → [One Big FFN]                → x'
MoE:      x → [Attention] → [Router → 6 of 256 tiny FFNs] → x'
                               ↑                      
                        Each token gets a custom 
                        blend of specialists
        
Same compute. 256× the knowledge capacity.
```