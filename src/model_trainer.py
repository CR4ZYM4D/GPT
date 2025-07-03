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

with open(train_set_path, 'r') as f:

    # read the data of the text file
    text = f.read()
    text = "Hello World How are you"
    # generate random points to make the text until there as the input_prompt incomplete wrods can help the model generalize better(hopefully)
    promptlines = torch.randint(len(text)//5, len(text)//3,  size = (4,)).tolist()

    # get the text prompts in a tensor batch
    input_tokens = torch.cat([model.config.tokenizer(text[:prompt], padding = 'max_length', max_length = model.max_sequence_length, return_tensors = 'pt')['input_ids'] for prompt in promptlines], dim = 0)
    output_tokens = torch.cat([model.config.tokenizer(text[1:], padding = 'max_length', max_length = model.max_sequence_length, return_tensors = 'pt')['input_ids']for _ in range(len(promptlines))], dim = 0)

# input_tokens, output_tokens = input_tokens.to(device), output_tokens.to(device)


print(input_tokens, output_tokens, sep = "\n\n")

print(input_tokens.shape, output_tokens.shape)
  #  print(i[0, 2:])

logits, loss = model.forward(input_tokens, output_tokens, device)

print(loss)
