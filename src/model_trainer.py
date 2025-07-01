import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from model_architecture import GPTModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/model.pkl"
train_set_path = "./gpt/dataset/training data.txt"

# print(device)

model = torch.load(model_path) if os.path.exists(model_path) else GPTModel()

# model = model.to(device)



with open(train_set_path, 'r') as f:

    eos_token = model.config.tokenizer.encode("<|endoftext|>", add_special_tokens = False, padding = False, return_tensors = 'pt')
    texts = ["hello world"] * 4
    all_tokens = []
    for text in texts:
        tokens = model.config.tokenizer.encode(text, add_special_tokens = True, padding = 'max_length', max_length = 1023, return_tensors = 'pt')
        tokens = torch.cat((tokens, eos_token), dim = -1)
        print(tokens.shape)
        all_tokens.append( tokens)
    
    all_tokens = torch.cat([tokens for tokens in all_tokens], dim = 0)

    print(all_tokens)
    print(all_tokens.shape)

