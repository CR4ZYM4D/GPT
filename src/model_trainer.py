import torch
import torch.nn as nn
import deepspeed
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity

from model_architecture import GPTModel

from typing import List

from tqdm import tqdm

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/"
train_set_path = "./gpt/dataset/subset"
deepspeed_config_path = "./gpt/deepspeed_config.json"

print(device)

model = GPTModel()

model, optimizer, _, _ = deepspeed.initialize(
	model = model,
	model_parameters = model.parameters(),
	config = deepspeed_config_path
)

# initializing summary writer

summary_writer = SummaryWriter(log_dir = "./gpt/logs")

# tokenizing input and target sequences

def tokenize_sequences(input_texts: List[str], target_texts: List[str]):

    # so that we can send one of the complete sequences for four of the text sequences of the same file to reduce memory overhead
	multiplier = len(input_texts)//len(target_texts)
        
        # safety for error handling
	assert len(input_texts) % len(target_texts) == 0 and multiplier >= 1 , "invalid input to target sequences to train"

	tokenizer = model.module.config.tokenizer if hasattr(model, 'module') else model.config.tokenizer

	max_sequence_length = model.module.max_sequence_length if hasattr(model, 'module') else model.max_sequence_length

        # convert to tokens
	input_tokens = torch.cat([tokenizer(input_text, padding = "max_length", max_length = max_sequence_length, truncation=True, 
                                return_tensors = 'pt')['input_ids'] for input_text in input_texts],
                                dim = 0)

	target_tokens = torch.cat([tokenizer(target_text, padding = "max_length", max_length = max_sequence_length, truncation=True,
                        return_tensors = 'pt')['input_ids'].repeat(multiplier, 1) for target_text in target_texts],
                        dim = 0)
	
	return input_tokens, target_tokens

# adding checkpoints to save vRAM and cache
class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        def custom_forward(*inputs):
            return self.block(*inputs)
        return checkpoint(custom_forward, x, use_reentrant = False)

decoder_blocks = model.module.decoder if hasattr(model, "module") else model.decoder
new_blocks = [CheckpointBlock(b) for b in decoder_blocks]
if hasattr(model, "module"):
    model.module.decoder = nn.Sequential(*new_blocks)
else:
    model.decoder = nn.Sequential(*new_blocks)

# creating profiler to track things

with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
			 schedule = torch.profiler.schedule(wait = 1, warmup = 1, active = 3, repeat = 1),
			 on_trace_ready = torch.profiler.tensorboard_trace_handler('./gpt/profiler'),
			 record_shapes = True,
			 profile_memory = True, 
			 with_stack = True) as prof:

	# iterating through each subset

	for j in tqdm(range(21), desc = "subset number"):

		# loading the subset files and checking if model has used it for training

		src_directory = f"{train_set_path}{j}"

		with open("./gpt/dataset/completed_subsets.txt", 'r') as f:

			completed_subsets = f.readlines()

			if src_directory in completed_subsets:

				continue 

		directory_files = os.listdir(src_directory)

		num_files = len(directory_files)
	
		# initializing loss and perplexity for model performance tracking

		subset_loss = 0.0

		subset_perplexity = 0.0

		subset_avg_loss = []

		subset_avg_perplexity = []

		# iterating through subset in batches of 8

		for i in tqdm(range(0, num_files, 128), leave = False):

			batch_files = directory_files[i: i + 128]

			input_texts = []

			target_texts = []
		
			for file in batch_files:

				with open(f"./gpt/dataset/subset{j}/{file}", 'r') as f:

					text = f.read()
				
					indices = torch.randint(low = len(text)//30, high = len(text)-1, size = (4,)).tolist()
				
					input_texts.extend([text[:index+1] for index in indices])

					target_texts.extend([text])


			# setting the model in training mode

			model.train()

			model.zero_grad()

			# tokenizing input and target texts

			input_tokens, target_tokens = tokenize_sequences(input_texts, target_texts)

			input_tokens = input_tokens.to(model.device)
			
			target_tokens = target_tokens.to(model.device)

			target_tokens = target_tokens[:, 1:] # removing the first token as the model starts prediction from second token

			pad_id = model.module.pad_token_idx if hasattr(model, "module") else model.pad_token_idx

			pad_tensor = torch.full((512, 1), pad_id, dtype=torch.long, device=target_tokens.device)

			target_tokens = torch.cat((target_tokens, pad_tensor), dim=1)

			# using autocast for mixed precision training

			logits, loss = model.forward(input_tokens, target_tokens)

			# updating loss and perplexity

			subset_loss += loss.item()

			subset_perplexity += torch.exp(loss).item()

			subset_avg_loss.append(subset_loss/ (i//128 +1))

			subset_avg_perplexity.append(subset_perplexity/ (i//128 +1))

			# logging to tensorboard

			summary_writer.add_scalar(f"subset {j} average loss", subset_avg_loss[-1], i//128 + 1)

			summary_writer.add_scalar(f"subset {j} average perplexity", subset_avg_perplexity[-1], i//128 + 1)

			# scaling the loss

			model.backward(loss)

			model.clip_grad_norm(1.0)

			model.step()

			prof.step()

		# updating that the subset has been trained on

		with open("./gpt/dataset/completed_subsets.txt", 'w') as f:

			f.write(f"{src_directory}\n")

		# saving model for backup

		model.save_checkpoint(model_path, tag = f"subset{j}")

		# logging total loss and perplexity of each subset to tensorboard

		summary_writer.add_scalar(f"model total loss", subset_loss, j+1)

		summary_writer.add_scalar(f"model total perplexity", subset_perplexity, j+1)

		print(f"Subset {j} completed with total loss: {subset_loss} and total perplexity: {subset_perplexity}")

# closing summary writer

summary_writer.flush()

summary_writer.close()