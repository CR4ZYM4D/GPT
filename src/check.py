import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

tensor = torch.randn((4,8,24))

print(tensor)

max = (torch.argmax(tensor, dim = -1))

print(max, max.shape, sep = '\n')