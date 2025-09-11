import os
import time
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
)
from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation

model_name = "tiiuae/falcon-rw-1b"
dataset_name = "HuggingFaceH4/ultrachat_200k"
dataset_split = "train_sft"
num_calibration_samples = 256
max_seq_length = 512

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
dispatch_for_generation(model)

# Pipeline for initial inference
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

start = time.time()
initial_output = pipe("The future of AI is", max_new_tokens=50)
end = time.time()
initial_time = end - start

# Save full model before quantization
save_dir_full = model_name.split("/")[-1] + "-full"
model.save_pretrained(save_dir_full, safe_serialization=True)
tokenizer.save_pretrained(save_dir_full)

# Load and preprocess dataset
dataset = load_dataset(dataset_name, split=f"{dataset_split}[:{num_calibration_samples}]")
dataset = dataset.shuffle(seed=42)

def preprocess(example):
    text = ""
    for msg in example["messages"]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"{role}: {content}\n"
    return {"text": text}

dataset = dataset.map(preprocess, batched=False)

def tokenize_sample(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False
    )

dataset = dataset.map(tokenize_sample, remove_columns=dataset.column_names, batched=False)

# Quantization setup
recipe = AWQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

oneshot(
    model=model,
    dataset=dataset,
    recipe=recipe,
    max_seq_length=max_seq_length,
    num_calibration_samples=num_calibration_samples
)

# Save compressed model
save_dir_quant = model_name.split("/")[-1] + "-awq"
model.save_pretrained(save_dir_quant, safe_serialization=True)
tokenizer.save_pretrained(save_dir_quant)

# Reload quantized model for inference
quant_model = AutoModelForCausalLM.from_pretrained(save_dir_quant, device_map="auto")
quant_tokenizer = AutoTokenizer.from_pretrained(save_dir_quant)

dispatch_for_generation(quant_model)
model.eval() 

# Pipeline for quantized inference
quant_pipe = pipeline("text-generation", model=quant_model, tokenizer=quant_tokenizer)

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

# Print results
print("Initial Output:", initial_output[0]["generated_text"])
print("Initial inference time:", initial_time, "seconds")
print("Quantized Output:", quant_output[0]["generated_text"])
print("Quantized inference time:", quant_time, "seconds")
print("Original model size:", initial_size, "MB")
print("Quantized model size:", quantized_size, "MB")
