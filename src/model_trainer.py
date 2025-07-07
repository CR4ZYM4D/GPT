import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
from tqdm import tqdm

from model_architecture import GPTModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/model.pkl"
train_set_path = "./gpt/dataset/subset"

print(device)

model = torch.load(model_path) if os.path.exists(model_path) else GPTModel()

model = model.to(device)

for j in tqdm(range(21), desc = "subset number"):

	src_directory = f"{train_set_path}{j}"

	with open("./gpt/dataset/completed_subsets.txt", 'r') as f:

		if src_directory in f.read():

			continue 

	directory_files = os.listdir(src_directory)

	num_files = len(directory_files)
	
	subset_loss = 0.0

	trained_files = set()

	if(os.path.exists(f"./gpt/dataset/subset{j}_trained_files.txt")):

		with open(f"./gpt/dataset/subset{j}_trained_files.txt", 'r')as f:

			trained_files = set(f.read())

	for i in tqdm(range(0, num_files, 8), leave = False):

		batch_files = directory_files[i: i+8]

		if batch_files not in trained_files:
	
			count= 0

			input_texts = []

			target_texts = []
		
			for file in batch_files:

				with open(file, 'r') as f:

					text = f.read()
				
					indices = torch.randint(low = len(text)//30, high = len(text)-1, size = (4,)).tolist()
				
					input_texts = [text[:index+1] for index in indices] if count == 0 else input_texts.append([text[:index+1] for index in indices])

					target_texts = [text[1: ]] if count == 0 else target_texts.append(text[1: ])

					count += 1

			loss = model.train(input_texts, target_texts)

			torch.save(model, model_path)

			trained_files.update(batch_files)

			subset_loss += loss.item()

			with open(f"./gpt/dataset/subset{j}_trained_files.txt", 'w')as f:

				f.write(f"{file}\n" for file in batch_files)

	print(f"subset number {i} completed, loss in subset is: {subset_loss}")

	with open("./gpt/dataset/completed_subsets.txt", 'w') as f:

		f.write(f"{src_directory}\n")

	