import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# Load base model and run initial inference
model_name =  "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
prompt = "The future of AI is"

def benchmark(model, tokenizer, prompt):
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

# Quantize the model with BitsandBytes
bnb_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_compute_dtype=torch.bfloat16,
   bnb_4bit_use_double_quant=True,
   bnb_4bit_quant_storage=torch.bfloat16,
)

quant_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto")

quant_tokenizer = AutoTokenizer.from_pretrained(model_name)

quant_output, quant_time = benchmark(quant_model, quant_tokenizer, prompt)

# Print results
print("=== Full Model ===")
print(f" Output: {initial_output}")
print(f" Inference time: {initial_time:.4f} s")

print("\n=== Quantized Model ===")
print(f" Output: {quant_output}")
print(f" Inference time: {quant_time:.4f} s")

