import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import shutil
from tqdm import tqdm

from model_architecture import GPTModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/model.pkl"
train_set_path = "./gpt/dataset/subset"

print(device)

model = torch.load(model_path) if os.path.exists(model_path) else GPTModel()

model = model.to(device)

for i in tqdm(range(21), desc = "subset number"):

	current_directory  = os.path.join(f"{train_set_path}{i}")
	
	# returns list of folders in each subset
	current_directory_folders = os.listdir(current_directory)

	# moving all files from odd indexed folders to even indexed ones

	for i in tqdm(range(1,len(current_directory_folders), 2), desc = f"directory{i} and {i-1}", leave=False):

		destination_folder = os.path.join(current_directory, current_directory_folders[i-1])

		src_folder = os.path.join(current_directory, current_directory_folders[i])

		for item in os.listdir(src_folder):

			destination_path = os.path.join(destination_folder, item)

			src_path = os.path.join(src_folder, item)

			shutil.move(src_path, destination_folder)
