# AWQ Quantization with LLM Compressor

This guide shows how to quantize a model using **Activation-Aware Weight Quantization (AWQ)** with the [LLM Compressor](https://github.com/vllm-project/llm-compressor) toolkit.  
AWQ protects ~1% of the most important weight channels to reduce quantization error and improve performance when compared to uniform quantization.
