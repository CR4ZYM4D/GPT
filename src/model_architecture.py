import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_config import GPT1ModelConfig, GPT2ModelConfig, ModelBlockConfig
from model_components import DecoderBlock, TokenEmbeddings, PositionalEncodings, LayerNorm

default_block_config = ModelBlockConfig()

default_tokenizer = AutoTokenizer.from_pretrained("./gpt/models/tokenizer")

default_model_config = GPT1ModelConfig(default_block_config, default_tokenizer)

class GPTModel(nn.Module):

    def __init__(self, config: GPT2ModelConfig | GPT1ModelConfig = default_model_config):
        
        super().__init__()

        self.config = config
        self.vocab_size = len(self.config.tokenizer)
        self.pad_token_idx = self.config.tokenizer.convert_tokens_to_ids(self.config.tokenizer.pad_token)
        self.eos_token_idx = self.config.tokenizer.convert_tokens_to_ids(self.config.tokenizer.eos_token)
        self.embedding_dimension = self.config.block_config.embedding_dimension
        self.max_sequence_length = self.config.block_config.max_sequence_length

        self.input_embeddings = TokenEmbeddings(self.vocab_size, 
                                                self.embedding_dimension,
                                                self.pad_token_idx)

        self.positonal_encodings = PositionalEncodings(self.max_sequence_length, 
                                                       self.embedding_dimension)

        self.decoder = nn.Sequential(*[DecoderBlock(self.config.block_config, self.vocab_size, 
                                                   self.pad_token_idx) for _ in range(self.config.num_layers)])

        self.final_layer_norm = LayerNorm(self.embedding_dimension)

        self.vocab_layer = nn.Linear(self.embedding_dimension, self.vocab_size, device = 'cuda')

        self.optimizer = optim.AdamW(self.parameters(), lr = 1e-3)

        print(self.config.tokenizer.special_tokens_map)
        print(self.vocab_size)
        print(self.max_sequence_length)

    def forward(self, x: torch.Tensor, targets: torch.Tensor = None, device: str = 'cuda'):

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
        logits = self.vocab_layer(x)

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

        x = self.config.tokenizer(x, padding = "max_length", max_length = self.max_sequence_length, truncation=True,
                                return_tensors = 'pt')['input_ids'].to(device = 'cuda')
        
        result = torch.clone(x)

        self.eval()

        with torch.no_grad():

            # shape of x = batch_size x sequence_length
        
            final_token = torch.where(x[0, :] == self.eos_token_idx)

            final_token = final_token[0].item() - 1

            next_index = None

            while final_token < self.max_sequence_length and next_index != self.eos_token_idx:

                logits, loss = self.forward(x)

                logits = torch.softmax(logits, dim = -1)

                # get probabilites of the final_index
                logits = logits[0, final_token, :]                

                next_token = torch.argmax(logits.view(1, self.vocab_size), dim = -1)

                result[0, final_token] = next_token[0].item()

                result[0, final_token+1] = self.eos_token_idx

                final_token += 1

        return self.config.tokenizer.decode(result)