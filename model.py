import torch
import torch.nn as nn
import torch.nn.functional as F

# using GPU if available

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# function to read a file and return the set of all characters in it for tokenization

def readTextFile(path: str):

    with open(path, 'r', encoding = 'UTF-8') as f:

        text = f.read()

    characters = sorted(set(text))

    return text, characters

text, characters = readTextFile('./src/wizardOfOz.txt')

#lambda functions to encode and decode  the text

stringToNum = {ch: i for i, ch  in enumerate(characters)}

numToString = {i: ch for i,ch in enumerate(characters)}

encode = lambda s: [stringToNum[c] for c in s]

decode = lambda s: [numToString[n] for n in s]

#converting the text into a tensor

data = torch.tensor(encode(text))

