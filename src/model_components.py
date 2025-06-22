import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

# class for the token embeddings table. Takes in vocab_size and embedding/vector_dimension and return a table
# of embeddings of each token index in the vocab. This embedding essentially helps the model understand the grammatical 
# meaning of that token/word/character in the current context/sequence

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, embedding_dimension: int, pad_idx: Optional[int], device: str):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.pad_idx = pad_idx
        self.device = device

        self.embeddings = nn.Embedding(self.vocab_size, 
                                       self.embedding_dimension, 
                                       padding_idx= self.pad_idx,
                                       device= device)
    
    def forward(self, token_ids: torch.Tensor):

        return self.embeddings(token_ids).to(self.device) 
    

# class for the positional encodings

class PositionalEncodings(nn.Module):

    def __init__(self, max_sequence_length: int, embedding_dimension: int, device: str):
        super().__init__()

        self.sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.device = device

        self.encodings = nn.Embedding(self.sequence_length, self.embedding_dimension, device = device)

    def forward(self, input_text: torch.Tensor):

        return self.encodings(input_text).to(self.device)
    
