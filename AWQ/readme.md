# AWQ Quantization with LLM Compressor

This guide shows how to quantize a model using **Activation-Aware Weight Quantization (AWQ)** with the [LLM Compressor](https://github.com/vllm-project/llm-compressor) toolkit.  
AWQ protects ~1% of the most important weight channels to reduce quantization error and improve performance when compared to uniform quantization.

# Running the AWQ Quantization Script

## 1. Clone or prepare your repository
Make sure your code (e.g., `main.py`) is saved inside your project folder.

## 2. Create and activate a Python environment (recommended)

```bash
# Create a new environment (optional, but recommended)
python3 -m venv .venv
source .venv/bin/activate   # for Linux/macOS
# OR
.venv\Scripts\activate      # for Windows PowerShell

```

## 3. Install dependencies

The script needs PyTorch, Transformers, Datasets, and LLM Compressor.

```bash
pip install torch transformers datasets llmcompressor
```

⚡ If you are using CUDA (GPU), make sure to install the GPU version of PyTorch:
👉 [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)


## 4. Run the script

Simply run your script with Python:

```bash
python main.py
```


## 5. Output

The script will:

- Load a model (meta-llama/Meta-Llama-3-8B-Instruct)
- Apply AWQ quantization
- Save the quantized model in a folder like:

```bash
Meta-Llama-3-8B-Instruct-awq-asym/
```

- You will also see sample generations printed in your terminal for verification.

  
✅ After this, you can reload your saved model (from Meta-Llama-3-8B-Instruct-awq-asym/) in any script using transformers.
