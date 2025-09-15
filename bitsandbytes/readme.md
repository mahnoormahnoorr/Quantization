# BitsAndBytes Quantization with Transformers

This example demonstrates quantizing the **OPT-125M** model using the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 4-bit quantization API in the 🤗 `transformers` library. Bitsandbytes uses runtime quantization, which compresses weights to 4-bit on-the-fly during inference to save memory, but these quantized weights cannot be saved to disk.

## Getting Started

Note, these examples are written for LUMI. If you want to use Puhti or Mahti, make sure to change the module and request for resources in the approriate way for each environment. 

All of the libraries needed to run this example (transformers, bitsandbytes, accelerate) are covered by the CSC preinstalled PyTorch module.

To run the example scripts, you can use a GPU interactively:
```bash
# Replace with your own project
srun --account=project_xxxxxxxx --partition=small-g --ntasks=1 --cpus-per-task=7 --gpus-per-node=1 --mem=16G --time=00:30:00 --nodes=1 --pty bash

# Load the module
module purge
module use /appl/local/csc/modulefiles
module load pytorch

python3 bitsandbytes_quantization.py
```

You can also submit a batch job. If you're quantizing a larger model, a batch job is recommended. Remember to change the project number and path.

```bash
sbatch run_bitsandbytes_quantization.sh
```
The script will quantize the OPT-125M model to nf4 or NormalFloat 4-bit, introduced to use with QLoRA technique, a parameter efficient fine-tuning technique. It can be used with QLoRA for fine-tuning, or without just for reducing model size.

## Output Includes

- Generated text before and after quantization.
- Inference time comparison.

## Notes

- The current scripts use OPT-125M for fast experimentation. You can replace model_name with a larger model.
- For large models, device_map="auto" lets 🤗 Accelerate handle placement across GPUs.
- You can easily change the quantization datatype in the `BitsAndBytesConfig` inside the script.
