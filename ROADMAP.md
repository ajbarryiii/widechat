# Architectural Roadmap: Sequential Depth vs. Parallel Breadth

## Objective
Investigate the tradeoffs between sequential depth and parallel breadth in the nanochat architecture while keeping the total parameter count relatively constant (within $\pm 10\%$). The goal is to maximize GPU utilization by processing parallel blocks concurrently using batched operations, avoiding any Python loop overhead.

## 1. The Batched Parallel Architecture
The proposed architecture restructures the model to process $B$ parallel branches simultaneously over $D$ sequential layers.

**Data Flow:**
1.  **Token Embedding:** `x_emb = norm(wte(tokens))` $\rightarrow$ `(Batch, Seq, n_embd)`
2.  **Split/Broadcast (Linear In):** 
    *   `x_branches = linear_in(x_emb)` $\rightarrow$ `(Batch, Seq, B * n_embd)`
    *   Reshape to: `(Batch, Seq, B, n_embd)`
3.  **Sequential Depth of Truly Parallel Blocks:** 
    *   For $i$ in range $D$:
        *   `x_branches = BatchedParallelBlock_i(x_branches)`
    *   *Mechanism:* Uses custom `BatchedLinear` layers and reshapes the `(Batch, Seq, B, ...)` tensors into `(Batch * B, Seq, ...)` before passing them to Flash Attention 3. This guarantees all branches execute in a single, perfectly parallelized CUDA kernel.
4.  **Collect (Linear Out):** 
    *   Reshape back to: `(Batch, Seq, B * n_embd)`
    *   `x = linear_out(x_branches)` $\rightarrow$ `(Batch, Seq, n_embd)`
    *   *Optional:* Add residual skip connection from the input embedding (`x = x + x_emb`).
5.  **Output:** `logits = lm_head(norm(x))`

## 2. Parameter Math & Target Configurations
With $B$ branches and a depth of $D$, the parameter math changes significantly from the $12$-layer sequential baseline.
*   **Baseline Block Size:** $12 \times 768^2 \approx 7.07$M params.
*   **Baseline Total (12x1):** $12 \times \text{Block Size} \approx 84.9$M params (excluding embeddings and LM head).
*   **New `linear_in` & `linear_out` cost:** $2 \times (B \times 768^2)$
*   **New Parallel blocks cost:** $D \times B \times \text{Block Size}$

To stay within $\pm 10\%$ of the baseline, we will test the following Depth ($D$) x Breadth ($B$) configurations:

*   **Baseline:** 12 x 1 ($86.11$M | $+1.4\%$)
*   **Config A:** 6 x 2 ($87.29$M | $+2.8\%$)
*   **Config B:** 4 x 3 ($88.47$M | $+4.2\%$)
*   **Config C:** 3 x 4 ($89.65$M | $+5.6\%$)
*   **Config D:** 2 x 5 ($76.68$M | $-9.7\%$)
*   **Config E:** 2 x 6 ($92.01$M | $+8.3\%$)
*   **Config F:** 1 x 10 ($82.58$M | $-2.8\%$)

## 3. Codebase Implementation Strategy

### A. Configuration (`nanochat/gpt.py`)
*   Update `GPTConfig` to include `n_branches: int = 1` and redefine `n_layer` to represent the sequential depth ($D$).

### B. Batched Linear Module (`nanochat/gpt.py`)
*   Create `BatchedLinear(nn.Module)` with weight shape `(Branch, Out, In)`.
*   Implement the forward pass using `torch.einsum('bsbi, boi -> bsbo', x, self.weight)`.

### C. Batched Parallel Block (`nanochat/gpt.py`)
*   Rewrite `CausalSelfAttention` and `MLP` to operate on `(Batch, Seq, Branch, Features)`.
*   Use the new `BatchedLinear` for Q, K, V, and MLP projections.
*   **Flash Attention 3 Integration:** Reshape Q, K, V from `(Batch, Seq, Branch, Heads, HeadDim)` to `(Batch * Branch, Seq, Heads, HeadDim)` before calling `flash_attn_func`. Reshape back immediately after.

### D. The Macro Architecture (`nanochat/gpt.py`)
*   Update `GPT.forward` to implement the `linear_in` -> `[BatchedParallelBlocks]` -> `linear_out` flow.

### E. KV Cache Adjustments (`nanochat/engine.py` & `gpt.py`)
*   Update `KVCache` to pre-allocate for the `Batch * Branch` dimension trick: `(n_layers, batch_size * num_branches, seq_len, num_heads, head_dim)`.

### F. Optimizer Grouping (`nanochat/optim.py` & `gpt.py`)
*   Validate that `MuonAdamW` correctly processes the 3D weight tensors from `BatchedLinear`. Muon inherently operates on 2D matrices, so the grouping logic in `setup_optimizer` will need to reshape `(B, Out, In)` into `(B * Out, In)` for the orthogonalization step.

## 4. Execution Plan

1.  **Drafting:** Implement `BatchedLinear`, the tensor reshapes, and `BatchedParallelBlock` in `gpt.py`. Update `GPTConfig`.
2.  **Optimizer Refactor:** Update `setup_optimizer` to support 3D batched weights for Muon.
3.  **Unit Testing:** Validate forward/backward passes and ensure `num_scaling_params()` matches the theoretical targets for the $1\times10$ and $2\times6$ configurations.
4.  **Throughput Profiling:** Run training benchmarks comparing the $12\times1$ baseline against the $2\times5$ and $1\times10$ batched configurations to ensure the batched operations match or exceed baseline tokens/sec.
5.  **Small-Scale Convergence Sweeps:** Execute 1-2 billion token training runs across all configurations to analyze validation loss trends.
