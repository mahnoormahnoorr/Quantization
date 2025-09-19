import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTQConfig,
    pipeline,
)

# Load base model and run initial inference
model_name = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
prompt = "The future of AI is"

def benchmark(model, tokenizer, prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Warm-up run (to remove cold start effects)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=5)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end = time.time()

    elapsed_time = end - start
    decoded_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return decoded_text, elapsed_time

# Run benchmark on full model
initial_output, initial_time = benchmark(model, tokenizer, prompt)

# Save full model before quantization
save_dir_full = model_name.split("/")[-1] + "-full"
model.save_pretrained(save_dir_full, safe_serialization=True)
tokenizer.save_pretrained(save_dir_full)


# Quantize the model with GPTQ
gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",       # Use a standard text dataset for calibration
    tokenizer=tokenizer
)

quantized_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=gptq_config,
    device_map="auto")

save_dir_quant = model_name.split("/")[-1] + "-gptq-config"
# Move model to a CPU for saving
quantized_model.to("cpu")
quantized_model.save_pretrained(save_dir_quant, safe_serialization=True)

tokenizer.save_pretrained(save_dir_quant)

# Reload quantized model and test inference
quant_model = AutoModelForCausalLM.from_pretrained(save_dir_quant, device_map="auto")
quant_tokenizer = AutoTokenizer.from_pretrained(save_dir_quant)

quant_output, quant_time = benchmark(quant_model, quant_tokenizer, prompt)

# Compare model sizes
def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)  # MB

initial_size = get_folder_size(save_dir_full)
quant_size = get_folder_size(save_dir_quant)

# Print results
print("=== Full Model ===")
print(f" Output: {initial_output}")
print(f" Size: {initial_size:.2f} MB")
print(f" Inference time: {initial_time:.4f} s")

print("\n=== Quantized Model ===")
print(f" Output: {quant_output}")
print(f" Size: {quant_size:.2f} MB")
print(f" Inference time: {quant_time:.4f} s")
