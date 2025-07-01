import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from model_config import ModelConfig, ModelBlockConfig
from model_components import DecoderBlock, TokenEmbeddings, PositionalEncodings, LayerNorm

default_block_config = ModelBlockConfig()

default_tokenizer = AutoTokenizer.from_pretrained("./gpt/models/tokenizer")

default_model_config = ModelConfig(default_block_config, default_tokenizer)

class GPTModel(nn.Module):

    def __init__(self, config: ModelConfig = default_model_config):
        
        super().__init__()

        self.config = config
        self.vocab_size = len(self.config.tokenizer)
        self.pad_token_idx = self.config.tokenizer.convert_tokens_to_ids(self.config.tokenizer.pad_token)
        self.eos_token_idx = self.config.tokenizer.convert_tokens_to_ids(self.config.tokenizer.eos_token)
        self.device = self.config.block_config.device
        self.embedding_dimension = self.config.block_config.embedding_dimension
        self.max_sequence_length = self.config.block_config.max_sequence_length

        self.input_embeddings = TokenEmbeddings(self.vocab_size, 
                                                self.embedding_dimension,
                                                self.pad_token_idx,
                                                self.device)

        self.positonal_encodings = PositionalEncodings(self.max_sequence_length, 
                                                       self.embedding_dimension, 
                                                       self.device)

        self.decoder = nn.ModuleList([DecoderBlock(self.config.block_config, self.vocab_size, 
                                                   self.pad_token_idx) for _ in range(self.config.num_layers)])

        self.final_layer_norm = LayerNorm(self.embedding_dimension, self.device)

        self.vocab_layer = nn.Linear(self.embedding_dimension, self.vocab_size, device = self.device)

        print(self.config.tokenizer.special_tokens_map)
        print(self.vocab_size)
        print(self.max_sequence_length)

    def forward(self, x: torch.Tensor, final_index: int = 0, targets: torch.Tensor = None):

        input_embeds = self.input_embeddings(x)

        positions = self.positonal_encodings(torch.arange(start = 0, end = final_index+1, dtype = torch.int64, device = self.device))

        remaining_positions = torch.zeros((self.max_sequence_length - final_index, self.embedding_dimension), device = self.device)

        positions = torch.cat((positions, remaining_positions))

        x = input_embeds + positions

        x = self.decoder(x)

        x = self.final_layer_norm(x)

        logits = F.softmax(self.vocab_layer(x), dim = -1)

        if targets == None:

            loss = None

        else:

            # for when model is in training

            # logits shape = 32, 1024, 50260

            # targets shape = 32 x 1024 x 50260

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, x: torch.Tensor, final_index: int = 0):

        result = torch.clone(x[ : final_index+1])

        next_index = None

        while final_index < self.max_sequence_length and next_index != self.eos_token_idx:

            logits, loss = self.forward(x, final_index)

            next_token = torch.multinomial(logits[final_index - self.max_sequence_length, : ], num_samples = 1)

            final_index += 1

            result = torch.cat((result, next_token))

        return result
    