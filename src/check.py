import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

tensor = torch.randn((4,8))

print(tensor)

new_tensor = tensor.repeat(2,1)

print(new_tensor, new_tensor.shape, sep = '\n')