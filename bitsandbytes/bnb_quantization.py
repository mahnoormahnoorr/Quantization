import os
import time
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# Load base model and run initial inference
model_name =  "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

start = time.time()
initial_output = pipe("The future of AI is", max_new_tokens=50)
end = time.time()
initial_time = end - start

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

quant_pipe = pipeline("text-generation", model=quant_model, tokenizer=quant_tokenizer)

start = time.time()
quant_output = quant_pipe("The future of AI is", max_new_tokens=50)
end = time.time()
quant_time = end - start

# Print all results together
print("Initial Output:", initial_output[0]["generated_text"])
print("Initial inference time:", initial_time, "seconds")
print("Quantized Output:", quant_output[0]["generated_text"])
print("Quantized inference time:", quant_time, "seconds")
