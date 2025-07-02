import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

from model_architecture import GPTModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/model.pkl"
train_set_path = "./gpt/dataset/training data.txt"

# print(device)

model = torch.load(model_path) if os.path.exists(model_path) else GPTModel()

# model = model.to(device)

length = 0
output_prompts = []

with open(train_set_path, 'r') as f:

    # read the data of the text file
    text = f.read()

    # generate random points to make the text until there as the input_prompt incomplete wrods can help the model generalize better(hopefully)
    promptlines = torch.randint(len(text)//5, len(text)//3,  size = (4,)).tolist()

    # get the text prompts in a list
    input_prompts = [text[:prompt] for prompt in promptlines]

    output_prompts = [text[1:]] * len(promptlines)

output_prompts.append(text[1:])

input_tokens = model.config.tokenizer(input_prompts, padding = 'max_length', return_tensors = 'pt')['input_ids']
output_tokens = model.config.tokenizer(output_prompts, padding = 'max_length', return_tensors = 'pt')['input_ids']


print(input_tokens, output_tokens, sep = "\n\n")

print(input_tokens.shape, output_tokens.shape)
  #  print(i[0, 2:])
print()

