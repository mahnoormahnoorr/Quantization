# Quantization Examples with Transformers & LLM Compressor

This repository contains two practical examples of applying GPTQ quantization to LLMs.  
Both examples currently use the small **OPT-125M** model for demonstration, but the code is written so you can swap in larger models.

1. **`GPTQconfig.py`** — Uses Hugging Face `transformers` and [`GPTQConfig`](https://huggingface.co/docs/transformers/en/quantization/gptq) to quantize the **OPT-125M** model.
2. **`GPTQmodifier.py`** — Uses [LLM Compressor](https://github.com/vllm-project/llm-compressor) with a GPTQ recipe to quantize the **OPT-125M** model. 

---

## 📦 Installation

Note, these examples are written for LUMI. If you want to use Puhti or Mahti,
make sure to change the module and request for resources in the approriate way for each environment.

The CSC preinstalled PyTorch module covers most of the libraries needed to run these examples
(torch, transformers, datasets, accelerate). The rest can be installed on top of the module in a virtual environment.

### Load the module
```bash
module purge
module use /appl/local/csc/modulefiles
module load pytorch
```
### Create and activate a virtual environment using system packages
```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
```
### Install packages
```bash
pip install optimum
# Install GPTQmodel for the GPTQconfig example
pip install gptqmodel –no-build-isolation
# Install LLM Compressor for the GPTQmodifier example
pip install llmcompressor
```
## Usage

To run the example scripts, you can use a GPU interactively:
```bash
# Replace with your own project
srun --account=project_xxxxxxxx --partition=small-g --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=16G --time=00:30:00 --nodes=1 --pty bash

module purge
module use /appl/local/csc/modulefiles
module load pytorch

python3 GPTQmodifier.py
```

You can also submit a batch job. If you're quantizing a larger model, a batch job is recommended:
```bash
sbatch run_gptq_modifier.sh
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
- The current scripts use **OPT-125M** for fast experimentation. You can replace `model_name` with a larger model. In this case, you might want to disable saving the models.
- For large models, `device_map="auto"` lets 🤗 Accelerate handle placement across GPUs.
- The `GPTQConfig` path is simpler and integrates directly with Hugging Face pipelines, while the `GPTQModifier` path gives you more flexibility for research and custom recipes.
- Feel free to experiment with different values for ´num_calibration_samples´ and ´max_seq_lenght'.
