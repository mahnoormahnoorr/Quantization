from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

from transformers import pipeline

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
output = generator("The future of AI is", max_new_tokens=50)
print(output[0]["generated_text"])

import time

start = time.time()
output = pipe("The future of AI is", max_new_tokens=50)
end = time.time()

print("Inference time:", end - start, "seconds")

from transformers import GPTQConfig, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")

gptq_config = GPTQConfig(
    bits=4,
    dataset="c4",  # Use a standard text dataset for calibration
    tokenizer=tokenizer
)

from transformers import AutoModelForCausalLM

quantized_model = AutoModelForCausalLM.from_pretrained(
    "facebook/opt-125m",
    device_map="auto",  # Automatically balances model across GPU/CPU
    quantization_config=gptq_config
)

quantized_model.to("cpu")  # Move model to CPU before saving
quantized_model.save_pretrained("opt-125m-gptq")
tokenizer.save_pretrained("opt-125m-gptq")

from transformers import pipeline

model = AutoModelForCausalLM.from_pretrained("opt-125m-gptq", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("opt-125m-gptq")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
result = pipe("The future of AI is", max_new_tokens=50)
print(result[0]["generated_text"])

import time

start = time.time()
output = pipe("The future of AI is", max_new_tokens=50)
end = time.time()

print("Inference time:", end - start, "seconds")

import os

# Folder size in MB
def get_folder_size(path):
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / (1024 * 1024)


print("Quantized size:", get_folder_size("opt-125m-gptq"), "MB")

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model.save_pretrained("opt-125m-full")


print("Original size:", get_folder_size("opt-125m-full"), "MB")
