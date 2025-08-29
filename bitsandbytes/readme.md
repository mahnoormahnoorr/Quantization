# BitsAndBytes Quantization with Transformers

This guide demonstrates **quantizing a Hugging Face Transformer model** (here: [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m)) using the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 4-bit quantization API in the 🤗 `transformers` library.  
It shows how to:

# Running the BitsAndBytes Quantization Script

## 1. Clone or prepare your repository
Make sure your code (e.g., `bitsandbytes.py`) is saved inside your project folder.

## 2. Create and activate a Python environment (recommended)

```bash
# Create a new environment (optional, but recommended)
python3 -m venv .venv
source .venv/bin/activate   # for Linux/macOS
# OR
.venv\Scripts\activate      # for Windows PowerShell

```

## 3. Install dependencies

The script needs Transformers, PyTorch, and accelerate. 

```bash
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu118  # pick cu118 or cu121 depending on your CUDA
pip install "transformers>=4.41" "accelerate>=0.30" 
```

⚡ If you are using CUDA (GPU), make sure to install the GPU version of PyTorch: [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)


## 4. Run the script

Simply run your script with Python:

```bash
python bitsandbytes.py
```


## 5. Output

The script will:

- Load a model (facebook/opt-125m)
- Apply BitsAndBytes quantization
- Save the quantized model in a folder like: opt-125m-bnb/
- You will also see sample generations printed in your terminal for verification.

  
✅ After this, you can reload your saved model (from opt-125m-bnb/) in any script using transformers.
