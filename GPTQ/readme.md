# Quantization Examples with Transformers & LLM Compressor

This repository contains two practical examples of applying GPTQ quantization to LLMs.  
Both examples currently use the small **OPT-125M** model for demonstration, but the code is written so you can swap in larger models if you have the resources.

1. **`GPTQconfig.py`** — Uses Hugging Face `transformers` and [`GPTQConfig`](https://huggingface.co/docs/transformers/en/quantization/gptq) to quantize the **OPT-125M** model.
2. **`GPTQmodifier.py`** — Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a GPTQ recipe to quantize the **OPT-125M** model. 

---

## 📦 Installation

It is recommended to use a fresh Python environment.

```bash
# Create and activate a virtual environment using system packages (HPC-friendly)
python3 -m venv --system-site-packages .venv
# Activate the environment
source .venv/bin/activate
```

Install dependencies
```bash
pip install --upgrade torch transformers datasets optimum accelerate
```

Install GPTQmodel for GPTQconfig example
```bash
pip install gptqmodel --no-build-isolation
```

Install LLM Compressor for GPTQmodifier example
```bash
pip install llmcompressor
```

## `GPTQConfig.py`
- Uses Hugging Face 🤗 `transformers` with [`GPTQConfig`](https://huggingface.co/docs/transformers/en/quantization/gptq).
- Saves both the full-precision and quantized models. 
- Compares outputs, inference latency, and model size.

## `GPTQModifier.py`
- Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a `GPTQModifier` recipe.
- Runs explicit **calibration** on a subset of the [Ultrachat-200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) dataset.
- Saves both the full-precision and quantized models.
- Compares outputs, inference latency, and model size.
- Provides finer control over quantization schemes (e.g. `W4A16`, `ignore=["lm_head"]`).

## Output Includes
- Generated text before and after quantization.
- Inference time comparison.
- Model size (MB) before and after quantization.

## Notes
- The current scripts use **OPT-125M** for fast experimentation. Replace `model_name` with a larger model to test real-world efficiency gains. In this case, you might want to disable saving the models.
- For large models, prefer `device_map="auto"` to let 🤗 Accelerate handle placement across GPUs.
- The `GPTQConfig` path is simpler and integrates directly with Hugging Face pipelines, while the `GPTQModifier` path gives you more flexibility for research and custom recipes.

