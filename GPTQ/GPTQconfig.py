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

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

start = time.time()
initial_output = pipe("The future of AI is", max_new_tokens=50)
end = time.time()
initial_time = end - start

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

quant_pipe = pipeline("text-generation", model=quant_model, tokenizer=tokenizer)

start = time.time()
quant_output = quant_pipe("The future of AI is", max_new_tokens=50)
end = time.time()
quant_time = end - start

# Compare model sizes
def get_folder_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)  # MB

initial_size = get_folder_size(save_dir_full)
quantized_size = get_folder_size(save_dir_quant)

# Print all results together
print("Initial Output:", initial_output[0]["generated_text"])
print("Initial inference time:", initial_time, "seconds")
print("Quantized Output:", quant_output[0]["generated_text"])
print("Quantized inference time:", quant_time, "seconds")
print("Original model size:", initial_size, "MB")
print("Quantized model size:", quantized_size, "MB")
