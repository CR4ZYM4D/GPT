import os
import torch
import random
import math
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_architecture import GPTModel  

#CONFIG
CHECKPOINT_PATH = "./gpt/models/subset19/mp_rank_00_model_states.pt"
VALIDATION_SET_PATH = "./gpt/dataset/subset20"
LOG_DIR = "./gpt/validation_logs"
MAX_SEQUENCE_LENGTH = 1024
BATCH_SIZE = 24
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model architecture...")
model = GPTModel().to(DEVICE)

print(f"Loading checkpoint from {CHECKPOINT_PATH} ...")
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)

if 'module' not in checkpoint:
    raise KeyError("Expected 'module' key in checkpoint file.")

missing_keys, unexpected_keys = model.load_state_dict(checkpoint['module'], strict=False)

print("Model loaded successfully with strict=False.")
if missing_keys:
    print(f"Missing keys during load: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys during load: {unexpected_keys}")

model.eval()

# Tokenizer
tokenizer = model.config.tokenizer
pad_token_id = model.pad_token_idx

# Logging 
summary_writer = SummaryWriter(log_dir=LOG_DIR)

# Dataset Files 
directory_files = os.listdir(VALIDATION_SET_PATH)
random.shuffle(directory_files)
num_files = len(directory_files)

subset_loss = 0.0
subset_avg_loss = []
subset_avg_perplexity = []

# ==== Validation Loop ====
with torch.no_grad():
    for i in tqdm(range(0, num_files, BATCH_SIZE), desc="Validating subset 20"):
        batch_len = min(BATCH_SIZE, num_files - i)
        batch_files = directory_files[i:i + batch_len]

        input_texts = []
        target_texts = []

        for file in batch_files:
            with open(os.path.join(VALIDATION_SET_PATH, file), 'r', encoding='utf-8') as f:
                text = f.read()
                if len(text.strip()) == 0:
                    continue
                index = random.randint(len(text) // 30, len(text) - 1)
                input_texts.append(text[:index + 1])
                target_texts.append(text)

        if not input_texts:
            continue

        multiplier = len(input_texts) // len(target_texts)
        input_tokens = torch.cat([tokenizer(inp, padding="max_length", max_length=MAX_SEQUENCE_LENGTH,
                                            truncation=True, return_tensors='pt')['input_ids']
                                  for inp in input_texts], dim=0)
        target_tokens = torch.cat([tokenizer(tgt, padding="max_length", max_length=MAX_SEQUENCE_LENGTH,
                                             truncation=True, return_tensors='pt')['input_ids'].repeat(multiplier, 1)
                                   for tgt in target_texts], dim=0)

        input_tokens = input_tokens.to(DEVICE)
        target_tokens = target_tokens.to(DEVICE)

        # Shift targets left, pad final token
        target_tokens[:, :-1] = target_tokens[:, 1:]
        target_tokens[:, -1] = pad_token_id

        # Forward pass
        logits, loss = model(input_tokens, target_tokens)

        if not torch.isfinite(loss):
            print(f"Non-finite loss at batch {i // BATCH_SIZE + 1}. Skipping.")
            continue

        subset_loss += loss.item()
        avg_loss = subset_loss / (i // BATCH_SIZE + 1)
        avg_ppl = math.exp(avg_loss)

        subset_avg_loss.append(avg_loss)
        subset_avg_perplexity.append(avg_ppl)

        step = i // BATCH_SIZE + 1
        summary_writer.add_scalar("Validation avg loss", avg_loss, step)
        summary_writer.add_scalar("Validation avg perplexity", avg_ppl, step)

# Final Log
summary_writer.flush()
summary_writer.close()

print(f"\n Validation on subset 20 complete.")
print(f"Average Loss: {subset_avg_loss[-1]:.4f}")
print(f"Average Perplexity (exp(loss)): {subset_avg_perplexity[-1]:.2f}")
