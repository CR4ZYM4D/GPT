import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# print(device)

# tensor = torch.randn((1, 1, 8))

# print(torch.argmax(tensor.view((1, 8)), dim = -1))

# new_tensor = tensor

# print(new_tensor, new_tensor.shape, sep = '\n')

print(torch.randint(2, 10, (2,)).tolist())