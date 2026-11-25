# Cluster GPU Architecture

## Repository Layout

The `cluster-gpu` repository is a clean, standalone distribution containing the precompiled Cluster GPU toolchain, standard GPU/AI libraries in pure Cluster (`.cl` / `.zk`), runnable examples, and test suites:

```
cluster-gpu/
├── bin/
│   ├── cl                       # Unified GPU driver & Web UI controller binary
│   ├── cl2                      # Precompiled Cluster compiler binary
│   └── cluster-gpu -> cl        # Dedicated GPU controller symlink
├── cl -> bin/cl                 # Root executable symlink
├── lib/
│   └── runtime/
│       └── concurrency.o        # Compiled runtime support object
├── std/
│   ├── ai/                      # Pure Cluster AI & GPU modules (gpu.cl, gguf.cl, transformer.cl, etc.)
│   ├── ai.cl / .zk              # Standard AI module
│   ├── clm.cl                   # Standard CLM neural model module
│   └── tensor.cl / .zk          # Standard Tensor module
├── examples/                    # Standalone runnable GPU examples (01..05)
├── tests/                       # GPU VRAM, Qwen, Indra-OCR, and tensor test suites
├── models/
│   └── uploaded/                # Destination for dynamically uploaded user models
├── docs/                        # Hardware and ingestion documentation
└── README.md                    # Flagship repository documentation
```

---

## 1. Zero Host-RAM Memory Model

Unlike typical frameworks that require full model weights in CPU Host RAM during execution:
1. `gpu_vram_load_file` opens the model file and transfers weights across the PCIe bus directly into an OpenGL 4.3 Shader Storage Buffer Object (`GL_SHADER_STORAGE_BUFFER`).
2. The transient host buffer is immediately released.
3. As a result, **100% of the model parameters reside exclusively inside the GPU physical VRAM**.

---

## 2. Autoregressive Inference Pipeline

When a prompt is submitted via `/api/chat` in the Web UI or via `clm_chat` in Cluster code:
1. **Tokenization**: Words/subwords converted into vocabulary IDs via embedded BPE tokenizer.
2. **GPU Vector Compute**: Forward kernel dispatches across the 48 Fermi CUDA cores.
3. **Layer Execution**:
   - RMSNorm normalizes token embeddings.
   - Rotary Position Embeddings (RoPE) encode spatial token order.
   - Multi-Head Attention computes Q, K, V matrices and KV cache updates.
   - SwiGLU activation computes feed-forward network transformations.
4. **Sampling & Detokenization**: ArgMax / Top-P selects the next token ID and appends to output text stream.
5. **Autoregressive Loop**: Repeats until EOS or maximum context length is reached.
