import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_config import GPT1ModelConfig, GPT2ModelConfig, ModelBlockConfig
from model_components import DecoderBlock, TokenEmbeddings, PositionalEncodings, LayerNorm

import os

default_block_config = ModelBlockConfig()

default_tokenizer = AutoTokenizer.from_pretrained("./gpt/models/tokenizer")

version = int(os.environ.get("MODEL_CONFIG_VERSION", 0))  # default GPT-1

default_model_config = GPT1ModelConfig(default_block_config, default_tokenizer) if version == 0 else GPT2ModelConfig(default_block_config, default_tokenizer)

def sample_top_p(logits, p: float = 1.0):

    # logits are of dimensions [1 x 1 x vocab_size]

    sorted_logits, logits_index = torch.sort(logits, dim = -1, descending=True)

    probability_sum = torch.cumsum(sorted_logits, dim = -1) # take cumulative sum

    mask = probability_sum - sorted_logits > p  # mask where cumulative sum exceeds p to remove highly improbable logits
 
    sorted_logits[mask] = 0.0

    sorted_logits = sorted_logits.divide(sorted_logits.sum(dim = -1, keepdim=True))  # normalize to get probabilities

    next_token_index = torch.multinomial(sorted_logits, num_samples=1)  # sample from the distribution

    next_token_index = logits_index.gather(dim=-1, index=next_token_index)  # gather the original indices

    return next_token_index[0].item()

class GPTModel(nn.Module):

    def __init__(self, config: GPT2ModelConfig | GPT1ModelConfig = default_model_config):
        
        super().__init__()

        self.config = config
        self.vocab_size = len(self.config.tokenizer)
        self.pad_token_idx = self.config.tokenizer.convert_tokens_to_ids(self.config.tokenizer.pad_token)
        self.eos_token_idx = self.config.tokenizer.convert_tokens_to_ids(self.config.tokenizer.eos_token)
        self.embedding_dimension = self.config.block_config.embedding_dimension
        self.max_sequence_length = self.config.block_config.max_sequence_length
        self.temperature = self.config.temperature

        self.input_embeddings = TokenEmbeddings(self.vocab_size, 
                                                self.embedding_dimension,
                                                self.pad_token_idx)

        self.positonal_encodings = PositionalEncodings(self.max_sequence_length, 
                                                       self.embedding_dimension)

        self.decoder = nn.Sequential(*[DecoderBlock(self.config.block_config, self.vocab_size, 
                                                   self.pad_token_idx) for _ in range(self.config.num_layers)])

        self.final_layer_norm = LayerNorm(self.embedding_dimension)

        self.vocab_layer = nn.Linear(self.embedding_dimension, self.vocab_size)

        self.optimizer = optim.AdamW(self.parameters(), lr = 1e-4)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None):

        # getting device
        device = x.device

        # shape of x = batch_size x sequence_length
        input_embeds = self.input_embeddings(x)

        # shape of x = batch_size x sequence_length x embedding_dimension
        positions = self.positonal_encodings(torch.arange(start = 0, end = self.max_sequence_length, dtype = torch.int64, device = device))

        # shape of x = batch_size x sequence_length x embedding_dimension
        x = input_embeds + positions

        # shape of x = batch_size x sequence_length x embedding_dimension
        x = self.decoder(x)

        # shape of x = batch_size x sequence_length x embedding_dimension
        x = self.final_layer_norm(x)

        # shape of x = batch_size x sequence_length x vocab_size
        logits = self.vocab_layer(x.to(self.vocab_layer.weight.dtype))

        if targets == None:

            loss = None

        else:

            # for when model is in training

            # logits shape = batch_size x sequence_length x vocab_size

            # targets shape = batch_size x sequence_length

            # predicted_tokens shape = batch_size x vocab_size x sequence_length
            predicted_tokens = logits.permute(0, 2, 1)

            loss = F.cross_entropy(predicted_tokens, targets)

        return logits, loss
    
    def generate(self, x: str):

        device = next(self.parameters()).device

        x = self.config.tokenizer(x, padding = "max_length", max_length = self.max_sequence_length, truncation=True,
                                return_tensors = 'pt')['input_ids'].to(device)
        
        result = torch.clone(x)

        self.eval()

        with torch.no_grad():

            # shape of x = batch_size x sequence_length
        
            final_token_index = torch.where(x[0, :] == self.eos_token_idx)

            final_token_index = final_token_index[0].item()

            next_token = None

            while final_token_index < self.max_sequence_length-1 and next_token != self.eos_token_idx:

                logits, loss = self.forward(x)

                logits = torch.softmax(logits / self.temperature, dim = -1)

                # get probabilites of the final_index
                logits = logits[0, final_token_index, :]                

                next_token = sample_top_p(logits, p = 0.9)

                result[0, final_token_index] = next_token

                result[0, final_token_index+1] = self.eos_token_idx

                final_token_index += 1

        return self.config.tokenizer.decode(result[0], skip_special_tokens=True, 
                                            clean_up_tokenization_spaces=True), final_token_index