# Quantization Examples with Transformers & LLM Compressor

This repository contains two practical examples of applying **quantization** to large language models (LLMs):

1. **`GPTQconfig.py`** — Uses Hugging Face `transformers` and `GPTQConfig` to quantize the **OPT-125M** model. It compares inference time and model size before and after quantization.  
2. **`GPTQmodifier.py`** — Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a GPTQ recipe to quantize the **Meta-Llama-3-8B-Instruct** model. It performs calibration on a dataset, runs inference, and saves the quantized model.

---

## 📦 Installation

It is recommended to use a fresh Python environment.

```bash
# (Optional) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate      # Linux/macOS
# OR
.venv\Scripts\activate         # Windows PowerShell

# Install dependencies
pip install torch datasets llmcompressor
pip install optimum 
pip install gptq —no-build-isolation 
pip install --upgrade accelerate optimum transformers

```

## ▶️ Running the Scripts

## 1. Run GPTQconfig.py

This script:
- Loads the OPT-125M model. You can load your own model that needs to be quantized. 
- Applies 4-bit GPTQ quantization with calibration.
- Saves the quantized model (opt-125m-gptq) and compares it to the full-precision model.
- Measures inference speed and disk size.

  
Run it with:

```bash
python GPTQconfig.py
```

Output includes:
- Generated text before and after quantization.
- Inference time comparison.
- Model size (MB) before and after quantization.

## 2. Run GPTQmodifier.py

This script:
- Loads the Meta-Llama-3-8B-Instruct model. You can load your own model that needs to be quantized.
- Prepares a calibration dataset (HuggingFaceH4/ultrachat_200k). You can load data from your own code.
- Applies GPTQ quantization (4-bit, group size 128) using LLM Compressor.
- Tests sample generation.
- Saves the quantized model (Meta-Llama-3-8B-Instruct-W4A16-G128).
- Reloads the saved model and performs inference.

Run it with:

```bash
python GPTQmodifier.py
```


Output includes:
- Sample generations from the quantized model.
- A quick test pipeline (e.g., generating a poem about Helsinki).
