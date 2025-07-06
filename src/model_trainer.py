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

for i in tqdm(range(21), desc = "subset number"):

	src_directory = f"{train_set_path}{i}"

	directory_files = os.listdir(src_directory)

	num_files = len(directory_files)

	for i in range(0, num_files, 8):

		batch_files = directory_files[i: i+8]

		count= 0

		input_texts = []

		target_texts = []
		
		for file in batch_files:

			with open(file, 'r') as f:

				text = f.read()
				
				indices = torch.randint(low = len(text)//30, high = len(text)//3, size = (4,)).tolist()
				
				input_texts = [text[:index+1] for index in indices] if count == 0 else input_texts.append([text[:index+1] for index in indices])

				target_texts = [text[1: ]] if count == 0 else target_texts.append(text[1: ])

				count += 1

		loss = model.train(input_texts, target_texts)