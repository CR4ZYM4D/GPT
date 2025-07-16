import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from model_config import ModelBlockConfig

# class for the token embeddings table. Takes in vocab_size and embedding/vector_dimension and return a table
# of embeddings of each token index in the vocab. This embedding essentially helps the model understand the grammatical 
# meaning of that token/word/character in the current context/sequence

class TokenEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, embedding_dimension: int, pad_token_idx: Optional[int]):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dimension = embedding_dimension
        self.pad_token_idx = pad_token_idx

        self.embeddings = nn.Embedding(self.vocab_size, 
                                       self.embedding_dimension, 
                                       padding_idx= self.pad_token_idx, device = 'cuda')
    
    def forward(self, token_ids: torch.Tensor):

        return self.embeddings(token_ids) 
    

# class for the positional encodings

class PositionalEncodings(nn.Module):

    def __init__(self, max_sequence_length: int, embedding_dimension: int):
        super().__init__()

        self.sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension

        self.encodings = nn.Embedding(self.sequence_length, self.embedding_dimension, device = 'cuda')

    def forward(self, input_text: torch.Tensor):

        return self.encodings(input_text)
    
# class for layer normalization
    
class LayerNorm(nn.Module):

    def __init__(self, embedding_dimension: int):
        super().__init__()

        self.embedding_dimension = embedding_dimension
        self.layer = nn.LayerNorm(self.embedding_dimension, device = 'cuda')

    def forward(self, x: torch.Tensor):

        return self.layer(x)
    
# class for a single attention head

class AttentionHead(nn.Module):

    def __init__(self, max_sequence_length: int, head_dimension: int, dropout_fraction: Optional[float] = 0.4):
        super().__init__()

        self.seq_length = max_sequence_length
        self.head_dimension = head_dimension

        self.dropout = nn.Dropout(dropout_fraction)

        self.query_weights = nn.Linear(self.head_dimension, 
                                       self.head_dimension, bias = False, device= 'cuda')
        self.key_weights = nn.Linear(self.head_dimension, 
                                       self.head_dimension, bias = False, device= 'cuda')
        self.value_weights = nn.Linear(self.head_dimension, 
                                       self.head_dimension, bias = False, device= 'cuda')
        self.register_buffer('tril', torch.tril(torch.ones(max_sequence_length, max_sequence_length, 
                                                           dtype = torch.int16, device= 'cuda')))


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

        return self.dropout(output)

# class for Multi Head Self Attention

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, max_sequence_length: int, embedding_dimension: int, num_heads: int, dropout_fraction: Optional[float] = 0.4):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads

        self.dropout = nn.Dropout(dropout_fraction)

        assert self.embedding_dimension % self.num_heads == 0 ,"embedding dimension is not divisible by the number of heads"

        self.head_dimension = self.embedding_dimension // self.num_heads

        self.output_weights = nn.Linear(self.head_dimension * num_heads, self.embedding_dimension, bias = False,
                                         dtype = torch.float32, device = 'cuda')
        
        self.attention_heads =  nn.ModuleList([AttentionHead(self.max_sequence_length, 
                                                        #    self.embedding_dimension, 
                                                           self.head_dimension,  
                                                           dropout_fraction) for _ in range(self.num_heads)])
        
    def forward(self, x: torch.Tensor):

        # x is a tensor of dimensions [batch_size x sequence_length x embedding_dimension]

        b, s, v = x.shape

        # pass successive parts of x along the embedding_dimension, each divided into a tensor of head_dimension to each individual attention head
        # i.e. if a tensor has embedding dimension of 1024 and 8 heads, send the first 1024//8 = 128 embeddings (0..127) into the first head then the second
        # 128 (128..255) into the second head and so on....

        # breaking embedding_dimension of x into num_heads and head_dimension

        x = x.view(b, s, self.num_heads, self.head_dimension)

        # breaking of x into num_heads chunks with dimensions of [batch_size x sequence_length x 1 x head_dimension]
        x = x.chunk(chunks = self.num_heads, dim = 2)

        # passing each chunk into a separate attention head and then concatenating back all the results together to get a resulting tensor of dimensions
        # [batch_size x sequence_length x embedding_dimension]. (Since it is asserted that num_heads * head_dimension = embedding_dimension)

        attended_scores = [self.attention_heads[i](torch.squeeze(x[i], dim = 2)) for i in range(len(self.attention_heads))]

        attended_scores = torch.cat(attended_scores, dim = 2)
        
        # pass thorugh the linear layer to get the final contextualized tensors of dimensions
        # [batch_size x sequence_length x e,bedding_dimension]

        contextualized_embeddings = self.output_weights(attended_scores)

        return self.dropout(contextualized_embeddings)
    
# class for the feed forward block after the attention block

class FeedForward(nn.Module):

    def __init__(self, embedding_dimension: int):
        
        super().__init__()

        self.embedding_dimension = embedding_dimension

        self.intermediate_dimension = self.embedding_dimension * 4

        self.layer1 = nn.Linear(self.embedding_dimension, self.intermediate_dimension, dtype = torch.float32, device = 'cuda')

        self.layer2 = nn.Linear(self.intermediate_dimension, self.embedding_dimension, dtype = torch.float32, device = 'cuda')

    def forward(self, x: torch.Tensor):

        x = self.layer1(x)

        x = F.gelu(x, approximate = "tanh")

        output = self.layer2(x)

        return output
    
# class of a decoder block which contains all the blocks and will be stacked on top of each other

class DecoderBlock(nn.Module):

    def __init__(self, config: ModelBlockConfig, vocab_size:int, pad_token_idx: int):

        super().__init__()

        self.config = config

        self.embedding_dimension = self.config.embedding_dimension

        self.batch_size = self.config.batch_size

        self.max_sequence_length = self.config.max_sequence_length

        self.num_heads = self.config.num_heads

        self.dropout_fraction = self.config.dropout_fraction

        self.vocab_size = vocab_size

        self.pad_token_idx = pad_token_idx 

        self.layer_norm_1 = LayerNorm(self.embedding_dimension)

        self.multi_head_attention_block = MultiHeadSelfAttention(self.max_sequence_length, 
                                                                 self.embedding_dimension, 
                                                                 self.num_heads,  
                                                                 self.dropout_fraction)
        
        self.layer_norm_2 = LayerNorm(self.embedding_dimension)

        self.feed_forward_block = FeedForward(self.embedding_dimension)

    def forward(self, x:torch.Tensor):

        # passing the token tensors through the first layer norm
        x = self.layer_norm_1(x)

        # passing the normalized tensor through the multi head self attention mechansim  
        contextualized_embeddings = self.multi_head_attention_block(x)

        # skip connection between contextualized and input embeddings
        updated_embeddings = x + contextualized_embeddings

        # passing these embeddings through the second layer norm
        updated_embeddings = self.layer_norm_2(updated_embeddings)

        # passing them through the feed forward block
        final_embeddings = self.feed_forward_block(updated_embeddings)

        # skip connection between feed forward and updated_embeddings
        final_embeddings = final_embeddings + updated_embeddings

        return final_embeddings
        