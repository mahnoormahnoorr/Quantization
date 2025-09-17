# Quantization


This repository explores **quantization methods for Large Language Models (LLMs)**.  
Quantization is a key technique for reducing model size and inference cost, enabling LLMs to run efficiently on consumer hardware or limited GPU memory.

We provide examples and experiments for:

- **[BitsAndBytes (bnb)](https://github.com/mahnoormahnoorr/Quantization/tree/main/bitsandbytes)** – nf4 quantization using the Hugging Face integration.
- **[AWQ (Activation-aware Weight Quantization)](https://github.com/mahnoormahnoorr/Quantization/tree/main/AWQ)** – a method that preserves accuracy by considering activation statistics.
- **[GPTQ (Gradient Post-training Quantization)](https://github.com/mahnoormahnoorr/Quantization/tree/main/GPTQ)** – post-training quantization optimized for autoregressive transformers.

---

## 📖 What to Expect in This Repo

1. **Implementation Examples**  
   - Scripts for loading, quantizing, and saving models with each method.  
   - Examples include small models (`facebook/opt-125m`) so you can try things quickly, and notes for scaling to larger models.

2. **Benchmarks**  
   - Inference time comparisons before and after quantization.  
   - Model size reduction (disk footprint in MB/GB).  
   - VRAM usage snapshots where applicable.

3. **Guides & Utilities**  
   - Helper functions for measuring folder size, timing inference, and testing outputs.  
   - Notes on environment setup for GPU clusters (CUDA / PyTorch / bitsandbytes compatibility).

4. **Reproducibility**  
   - Each script is self-contained and documented.  
   - Expected output snippets are included in the `README` sections or script comments.

---

