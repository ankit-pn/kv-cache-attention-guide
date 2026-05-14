# Glossary

| Term | Definition |
|---|---|
| **Q** (Query) | Vector representing "what this token is looking for" (one-time use) |
| **K** (Key) | Vector representing "what information this token contains" (cached) |
| **V** (Value) | Vector representing "this token's actual content" (cached) |
| **W_Q, W_K, W_V** | Learned weight matrices that project embeddings into Q, K, V space |
| **W_O** | Output projection — mixes information across attention heads |
| **KV Cache** | Stored K and V tensors for all previous tokens at every layer |
| **Causal Mask** | Upper-triangular mask of -inf preventing tokens from attending to future tokens |
| **Self-Attention** | Each token attending to all previous tokens (including itself) |
| **Prefill** | Processing all prompt tokens in parallel, computing and caching K,V |
| **Decode** | Generating one token at a time using cached K,V |
| **MHA** (Multi-Head Attention) | All h heads have separate Q, K, V projections |
| **GQA** (Grouped-Query Attention) | Fewer KV heads than query heads |
| **MQA** (Multi-Query Attention) | Single shared K,V head across all query heads |
| **Attention Head** | One independent attention computation, sees d_k dimensions |
| **d_k** | Dimension per attention head (d_k = d / h) |
| **h** | Number of attention heads |
| **d** | Hidden dimension of the model |
| **LM Head** | Final linear projection from hidden state to vocabulary logits |
| **Softmax** | Normalizes attention scores into probability distribution summing to 1 |
| **Residual Connection** | Adding input to output (x + F(x)), preserves gradient flow |
| **Prefix Cache** | Stored KV cache for shared prompt prefixes across requests |
| **Block Table** | Mapping from logical token positions to physical GPU memory blocks |
| **PagedAttention** | OS-style virtual memory for KV cache management |
| **TTFT** (Time to First Token) | Latency from request to first generated token (prefill-dominated) |
| **ITL** (Inter-Token Latency) | Time between successive generated tokens (decode-dominated) |
