from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.awq import AWQModifier

import transformers
import torch

from llmcompressor.utils import dispatch_for_generation  # important for runtime hooks

# Select model and load it.
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# Select calibration dataset.
DATASET_ID = "mit-han-lab/pile-val-backup"
DATASET_SPLIT = "validation"

# Select number of samples. 256 samples is a good place to start.
# Increasing the number of samples can improve accuracy.
NUM_CALIBRATION_SAMPLES = 256
MAX_SEQUENCE_LENGTH = 512

# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)

def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            [{"role": "user", "content": example["text"]}],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


# Configure the quantization algorithm to run.
recipe = [
    AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
]

# Apply algorithms.
oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Confirm generations of the quantized model look sane.
print("\n\n")
print("========== SAMPLE GENERATION ==============")
input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to("cuda")
output = model.generate(input_ids, max_new_tokens=100)
print(tokenizer.decode(output[0]))
print("==========================================\n\n")


 #Save to disk compressed.
SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-awq-asym"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)

SAVE_DIR = "Meta-Llama-3-8B-Instruct-awq-asym"  # your folder name

tokenizer = AutoTokenizer.from_pretrained(SAVE_DIR)
model = AutoModelForCausalLM.from_pretrained(
    SAVE_DIR,
    device_map="auto",        # put weights on available GPU(s)
    torch_dtype="auto"        # match the saved dtype config
)

# For llmcompressor-quantized models, register runtime hooks:
dispatch_for_generation(model)

# quick test
prompt = "Hello my name is"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(out[0], skip_special_tokens=True))

from transformers import TextGenerationPipeline

pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer)
print(pipe("Write a short poem about Helsinki:", max_new_tokens=80)[0]["generated_text"])

