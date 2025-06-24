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

    def __init__(self, max_sequence_length: int, head_dimension: int, device: str, dropout_fraction: Optional[float] = 0.4):
        super().__init__()

        self.seq_length = max_sequence_length
        self.head_dimension = head_dimension
        self.device = device
        self.dropout = nn.Dropout(dropout_fraction)

        self.query_weights = nn.Linear(self.head_dimension, 
                                       self.head_dimension, bias = False, device = self.device)
        self.key_weights = nn.Linear(self.head_dimension, 
                                       self.head_dimension, bias = False, device = self.device)
        self.value_weights = nn.Linear(self.head_dimension, 
                                       self.head_dimension, bias = False, device = self.device)
        self.register_buffer('tril', torch.tril(torch.ones(max_sequence_length, max_sequence_length, 
                                                           dtype = torch.int16, device = self.device)))


    def forward(self, x: torch.Tensor):

        # we get a tensor x of dimensions [batch_size x sequence_length x head_dimension]

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

        attention_scores = (torch.matmul(query_results, key_results)) / (self.head_dimension**0.5)

        # apply attention mask to hide all future tokens to make training more realistic and replace all zeros with
        # minus infinity
         
        masked_attention_scores = torch.masked_fill(attention_scores, self.tril[:s, :s] == 0, float('-inf'))

        # apply softmax to get probability like fractional scores and make all minus infinities as zero
        
        attention_weights = F.softmax(masked_attention_scores, dim = -1, dtype = torch.float32)

        # perform matrix multiplication with value_results this results in a tensor of dimensions
        # [batch_size x sequence_length x head_dimension]

        output = torch.matmul(attention_weights, value_results)

        return output

# class for Multi Head Self Attention

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, max_sequence_length: int, embedding_dimension: int, num_heads: int, device: str, dropout_fraction: Optional[float] = 0.4):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.device = device
        self.dropout = nn.Dropout(dropout_fraction)

        assert self.embedding_dimension % self.num_heads == 0 ("embedding dimension is not divisible by the number of heads")

        self.head_dimension = self.embedding_dimension // self.num_heads

        self.output_weights = nn.Linear(self.head_dimension * num_heads, self.embedding_dimension, bias = False,
                                         device = self.device, dtype = torch.float32)
        
        self.attention_heads =  [AttentionHead(self.max_sequence_length, 
                                                           self.embedding_dimension, 
                                                           self.head_dimension, 
                                                           self.device, 
                                                           dropout_fraction) for _ in range(self.num_heads) for _ in range(self.num_heads)]
        
    def forward(self, x: torch.Tensor):

        # x is a tensor of dimensions [batch_size x sequence_length x embedding_dimension]

        b, s, v = x.shape

        # pass successive parts of x along the embedding_dimension, each divided into a tensor of head_dimension to each individual attention head
        # i.e. if a tensor has embedding dimension of 1024 and 8 heads, send the first 1024//8 = 128 embeddings (0..127) into the first head then the second
        # 128 (128..255) into the second head and so on....

        # breaking embedding_dimension of x into num_heads and head_dimension

        x = x.view(b, s, self.num_heads, self.head_dimension)

        # breaking of x into num_heads chunks with dimensions of [batch_size x sequence_length x 1 x head_dimension]
        x = x.chunk(chunk = self.num_heads, dim = 2)

        # passing each chunk into a separate attention head and then concatenating back all the results together to get a resulting tensor of dimensions
        # [batch_size x sequence_length x embedding_dimension]. (Since it is asserted the num_heads * head_dimension = embedding_dimension)

        attended_scores = torch.cat([attention_head(torch.squeeze(chunk, dim = 2)) for (chunk, attention_head) in (x, self.attention_heads)], dim = -1)

        # pass thorugh the linear layer to get the final contextualized tensors of dimensions
        # [batch_size x sequence_length x e,bedding_dimension]

        contextualized_embeddings = self.output_weights(attended_scores)

        return contextualized_embeddings
    



