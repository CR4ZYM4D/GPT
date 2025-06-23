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
    
# class for layer normalization
    
class LayerNorm(nn.Module):

    def __init__(self, embedding_dimension: int, device: str):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.device = device
        self.layer = nn.LayerNorm(self.embedding_dimension, device = self.device)

    def forward(self, x: torch.Tensor):

        return self.layer(x)
    
# class for a single attention head

class AttentionHead(nn.Module):

    def __init__(self, max_sequence_length: int, embedding_dimension: int, head_dimension: int, device: str, dropout_fraction: Optional[float] = 0.4):
        super().__init__()

        self.seq_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.device = device
        self.dropout = nn.Dropout(dropout_fraction)

        self.query_weights = nn.Linear(self.embedding_dimension, 
                                       self.head_dimension, bias = False, device = self.device)
        self.key_weights = nn.Linear(self.embedding_dimension, 
                                       self.head_dimension, bias = False, device = self.device)
        self.value_weights = nn.Linear(self.embedding_dimension, 
                                       self.head_dimension, bias = False, device = self.device)
        self.register_buffer('tril', torch.tril(torch.ones(max_sequence_length, max_sequence_length, 
                                                           dtype = torch.int16, device = self.device)))


    def forward(self, x: torch.Tensor):

        # we get a tensor x of dimensions [batch_size x sequence_length x embedding_dimension]

        b, s, v = x.shape

        # first multiply x with the query key and value weight matrices to get tensor of dimensions
        # [batch_size x sequence_length x head_dimension]

        query_results = self.query_weights(x)

        key_results = self.key_weights(x)

        value_results = self.value_weights(x)

        # multiply query_results with the transpose of key_results to get a tensor of dimensions
        # [batch_size x sequence_length x sequence_length] and divide it by square root of the embedding_dimension

        # transposing key_results matrix for the multiplication to get a tensor of dimensions
        # [batch_size x head_dimension x sequence_length]

        key_results = key_results.transpose(-2, -1)

        attention_scores = (torch.matmul(query_results, key_results)) / (self.embedding_dimension**0.5)

        # apply attention mask to hide all future tokens to make training more realistic and replace all zeros with
        # minus infinity
         
        masked_attention_scores = torch.masked_fill(attention_scores, self.tril[:s, :s] == 0, float('-inf'))

        # apply softmax to get probability like fractional scores and make all minus infinities as zero
        
        attention_weights = F.softmax(masked_attention_scores, dim = -1, dtype = torch.float32)

        # perform matrix multiplication with value_results this results in a tensor of dimensions
        # [batch_size x sequence_length x head_dimension]

        output = torch.matmul(attention_weights, value_results)

        return output



