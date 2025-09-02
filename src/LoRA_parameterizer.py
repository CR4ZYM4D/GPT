# class and functions to perform lora parameterization on the different transformer blocks
import torch 
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import os

from model_architecture import GPTModel

# create the class for LoRA parameterization of a general matrix like the vocab layer or the attention matrices

class LoRA_Parameterization(nn.Module):

    def __init__(self, in_features: int, out_features: int, rank: int = 1, alpha: int = 1, device: str = 'cpu'):
        
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.device = device

        # LoRA matrices are of type BA times scale factor where B and A are the intrinsic rank matrices and scaling is a normal scalar to maintain variance
        # dim(B) = [in_features x rank]
        # dim(A) = [rank x out_features]
        # scale factor = alpha / rank

        self.scale = self.alpha/self.rank

        self.lora_A = nn.Parameter(torch.zeros((self.rank, self.out_features)).to(self.device))

        # initialize LoRA B matrix as all zeros so that delta_W is zero at the start of training
        self.lora_B = nn.Parameter(torch.zeros((self.in_features, self.rank)).to(self.device))

        # initialize LoRA A matrix as Gaussian distribution with avg = 0, std = 1
        nn.init.normal_(self.lora_A, mean=0, std=1)

        # intialize an extra enabled flag to turn LoRA on/off at demand
        self.enabled = True

    def forward(self, original_weights: torch.Tensor):

        if self.enabled:

            #return the LoRA matrices with added to the original weights for performing the forward calls
            # adding a view call to the multiplied and scaled matrix for extra safety measure

            return original_weights + (torch.matmul(self.lora_B, self.lora_A)*self.scale).view(original_weights.shape)
        
        else:

            return original_weights


# function to actually parametrize each layer with its LoRA parametrization

def parametrize_layer(layer_tensor: nn.Module, device: str = 'cpu', rank: int = 1, alpha: int = 1):

    # LoRA can be applied to 2D matrices so skip everything 1D like LayerNorm

    if not hasattr(layer_tensor, "weight") or layer_tensor.weight.ndim !=2: 
        return


    if isinstance(layer_tensor, nn.Embedding):

        in_features, out_features = layer_tensor.num_embeddings, layer_tensor.embedding_dim

    else: 
        in_features, out_features = layer_tensor.weight.shape

    lora_parametrization = LoRA_Parameterization(in_features, out_features, rank, alpha, device)

    parametrize.register_parametrization(layer_tensor, "weight", lora_parametrization)

    return in_features,out_features

def apply_lora_to_model(model: nn.Module, device: str = 'cpu', rank: int = 1, alpha: int = 1):
   

    for name, module in model.named_modules():
        try:
           i,o = parametrize_layer(module, device=device, rank=rank, alpha=alpha)
           print(f"current LoRA matrix of B and A of sizes: [{i}, {rank}] and [{rank}, {o}] respectively")
           print(f"Applied Lora to layer: {name} of size: [{i}, {o}]")
        
        except Exception as e:
            # skip if the module can't be parametrized
            # print(f"Skipping {name}: {e}")
            continue

    return model

# testing if parameters are applied normally

if __name__ == "__main__":

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = GPTModel()

    model_state = torch.load(os.path.join('./gpt/models/subset19/', 'mp_rank_00_model_states.pt'), map_location = device)

    missing, unexpected = model.load_state_dict(model_state['module'], strict = False)

    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    model = model.to(device)

    apply_lora_to_model(model)