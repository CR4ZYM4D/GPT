import torch
import torch.optim as optim
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, ProfilerActivity

from model_architecture import GPTModel

from typing import List

from tqdm import tqdm

import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/model.pkl"
train_set_path = "./gpt/dataset/subset"

print(device)

# loading existing model or initializing one

model = torch.load(model_path) if os.path.exists(model_path) else GPTModel()

model = model.to(device)

optimizer = model.optimizer

# initializing summary writer

summary_writer = SummaryWriter(log_dir = "./gpt/logs")

# Grad Scaler 

scaler = GradScaler()

# adding checkpoints to save vRAM and cache
class CheckpointBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        def custom_forward(*inputs):
            return self.block(*inputs)
        return checkpoint(custom_forward, x)

model.decoder = nn.Sequential(*[CheckpointBlock(block) for block in model.decoder])

# tokenizing input and target sequences

def tokenize_sequences(input_texts: List[str], target_texts: List[str]):

    # so that we can send one of the complete sequences for four of the text sequences of the same file to reduce memory overhead
	multiplier = len(input_texts)//len(target_texts)
        
        # safety for error handling
	assert len(input_texts) % len(target_texts) == 0 and multiplier >= 1 , "invalid input to target sequences to train"

        # convert to tokens
	input_tokens = torch.cat([model.config.tokenizer(input_text, padding = "max_length", max_length = model.max_sequence_length, truncation=True, 
                                return_tensors = 'pt')['input_ids'] for input_text in input_texts],
                                dim = 0).to(device='cuda')

	target_tokens = torch.cat([model.config.tokenizer(target_text, padding = "max_length", max_length = model.max_sequence_length, truncation=True,
                        return_tensors = 'pt')['input_ids'].repeat(multiplier, 1) for target_text in target_texts],
                        dim = 0).to(device='cuda')
	
	return input_tokens, target_tokens
   
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

		for i in tqdm(range(0, num_files, 8), leave = False):

			batch_files = directory_files[i: i+8]

			input_texts = []

			target_texts = []
		
			for file in batch_files:

				with open(f"./gpt/dataset/subset{j}/{file}", 'r') as f:

					text = f.read()
				
					indices = torch.randint(low = len(text)//30, high = len(text)-1, size = (4,)).tolist()
				
					input_texts.extend([text[:index+1] for index in indices])

					target_texts.extend([text[1: ]])


			    # setting the model in training mode

				model.train()

				optimizer.zero_grad()

				# tokenizing input and target texts

				input_tokens, target_tokens = tokenize_sequences(input_texts, target_texts)

				# using autocast for mixed precision training

				with autocast(device_type='cuda'):

					logits, loss = model.forward(input_tokens, target_tokens)

				# updating loss and perplexity

				subset_loss += loss.item()

				subset_perplexity += torch.exp(loss).item()

				subset_avg_loss.append(subset_loss/ (i//8 +1))

				subset_avg_perplexity.append(subset_perplexity/ (i//8 +1))

				# logging to tensorboard

				summary_writer.add_scalar(f"subset {j} average loss", subset_avg_loss[-1], i//8 + 1)

				summary_writer.add_scalar(f"subset {j} average perplexity", subset_avg_perplexity[-1], i//8 + 1)

				# scaling the loss

				scaler.scale(loss).backward()

				scaler.step(optimizer)

				scaler.update()

				prof.step()

		# updating that the subset has been trained on

		with open("./gpt/dataset/completed_subsets.txt", 'w') as f:

			f.write(f"{src_directory}\n")

		# saving model for backup

		torch.save(model, model_path)

		# logging total loss and perplexity of each subset to tensorboard

		summary_writer.add_scalar(f"model total loss", subset_loss, j+1)

		summary_writer.add_scalar(f"model total perplexity", subset_perplexity, j+1)

		print(f"Subset {j} completed with total loss: {subset_loss} and total perplexity: {subset_perplexity}")

# closing summary writer

summary_writer.flush()

summary_writer.close()