import random
import mmap

import torch
import torch.nn as nn
import torch.nn.functional as F

# important constants to be used in the model
model_path = "./src/model.pkl"

block_size = 128 # size of a single word or a combination of words (we will refer to this a s a block)

batch_size = 12 # no. of said blocks or words that we will handle at once

vector_dimension = 512 # dimensions of each of the alphabet or token vector

dropout = 0.5

n_heads = 16 # no of attention heads

n_layers = 8 # no of block layers used 

max_sequence_length = 400 # max no of tokens that will be generated

learning_rate = 5e-2

max_iterations = 20000

train_step_iteration = 10

max_test_iterations = 200

test_iterations = 20

test_step_iterations = 100

# using GPU if available

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# function to read a file and return the set of all characters in it for tokenization

def readTextFile(path: str):

    with open(path, 'r', encoding = 'UTF-8') as f:

        text = f.read()

    characters = sorted(set(text))

    return characters

characters = readTextFile('./dataset/vocab.txt')

#lambda functions to encode and decode  the text

stringToNum = {ch: i for i, ch  in enumerate(characters)}

numToString = {i: ch for i,ch in enumerate(characters)}

encode = lambda s: [stringToNum[c] for c in s]

decode = lambda s: ''.join(numToString[n] for n in s)

#converting the text into a tensor

def getChunk(split):

    file_path = "./dataset/training data.txt" if split == "train" else "./dataset/testing data.txt"

    with open(file_path, "rb") as f:

        with mmap.mmap(f.fileno(), 0, access= mmap.ACCESS_READ) as mm:

            file_size = len(mm)

            start_pos = random.randint(0, file_size - block_size * batch_size -1)

            mm.seek(start_pos)

            block = mm.read(block_size * batch_size)

            decoded_block = block.decode(encoding = "utf-8", errors = "ignore").replace("\r", "")

            data = torch.tensor(encode(decoded_block), dtype = torch.long)

    return data

# function to split and return the training and testing batches

def getBatch(split = "train"):
    
    data = getChunk(split)

    index = torch.randint(high = len(data) - block_size -1, size = (batch_size,))
    
    x = torch.stack([data[i: i+block_size] for i in index])
    y = torch.stack([data[i+1: i+1+block_size] for i in index])

    x, y = x.to(device), y.to(device)

    return x, y

# class that performs the feed forward mechanism of the decoder block (class that normalizes the vectors, aplies Relu and again normalizes them)

class FeedForward(nn.Module):

    def __init__(self, vector_dimension):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Linear(vector_dimension, vector_dimension * 4),
            nn.ReLU(),
            nn.Linear(vector_dimension * 4, vector_dimension)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        return self.dropout(self.layer(x))
    
# class that defines the single head self attention

class AttentionHead(nn.Module):

    def __init__(self, dimension_head):
        super().__init__()

        self.key = nn.Linear(vector_dimension, dimension_head, bias = False)
        self.query = nn.Linear(vector_dimension, dimension_head, bias = False)
        self.value = nn.Linear(vector_dimension, dimension_head, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        batch, time, channel = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        att = (q @ k.transpose(-2, -1)) / (k.shape[-1] ** 0.5)
        att = att.masked_fill(self.tril[:time, :time] == 0 , float('-inf'))

        att = F.softmax(att, dim = -1)
        att = self.dropout(att)

        out = att @ v

        return out

# class that defines the multi head self attention mechanism

class MultiHeadAttention(nn.Module):

    def __init__(self, n_heads, dimension_head):
        super().__init__()

        self.attention_heads = nn.ModuleList([AttentionHead(dimension_head) for _ in range (n_heads)])

        self.projection_layer = nn.Linear(n_heads * dimension_head, vector_dimension)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = torch.cat([h(x) for h in self.attention_heads], dim = -1)
        out = self.dropout(self.projection_layer(out))

        return out

# class that defines a single decoder block of the model

class Block(nn.Module):

    def __init__(self, n_heads, vector_dimension):
        super().__init__()

        dimension_head = vector_dimension // n_heads

        self.attention = MultiHeadAttention(n_heads, dimension_head)

        self.feed_forward = FeedForward(vector_dimension)

        self.layer_norm_1 = nn.LayerNorm(vector_dimension)
        
        self.layer_norm_2 = nn.LayerNorm(vector_dimension)

    def forward(self, x):

        y = self.attention(x)

        x = self.layer_norm_1(x + y)

        y = self.feed_forward(x)

        x = self.layer_norm_2(x + y)

        return x


class GPTModel(nn.Module):

    # constructor

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(vocab_size, vector_dimension)

        self.positional_encodings = nn.Embedding(block_size, vector_dimension)

        self.layers = nn.Sequential(*[Block(n_heads, vector_dimension) for _ in range (n_layers)])

        self.final_layer_norm = nn.LayerNorm(vector_dimension)

        self.linear = nn.Linear(vector_dimension, vocab_size)

        self.apply(self.initWeights)

    # method to initialize the weights

    def initWeights(self, module):

        if isinstance(module, nn.Linear):

            torch.nn.init.normal_(module.weight, std = 0.02)
        
        elif isinstance(module, nn.LayerNorm):

            torch.nn.init.normal_(module.weight, std = 0.02)
    
    # method to return the token embedding of given index token and return the loss if there is a target set given

    def forward(self, index, targets= None):

        # getting the token embedding of the given token index
 
        batch, time = index.shape
        
        token_embedding = self.token_embeddings(index)

        positional_encoding =  self.positional_encodings(torch.arange(time, device = device))

        x = token_embedding + positional_encoding

        x = self.layers(x)

        x = self.final_layer_norm(x)

        logits = self.linear(x)

        if targets == None:

            loss=None

        else: 

            # making logits and targets of similar dimensions to calculate the loss
            batch, time, channels = logits.shape

            logits = logits.view(batch * time, channels)

            targets = targets.view(batch * time)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # method to generate the tokens

    def generate(self, index, max_sequence_length):

        result = torch.clone(index)
    
        for _ in range (max_sequence_length):

            logits, loss = self.forward(index)

            logits = logits[: , -1, :]

            probabilities = F.softmax(logits, dim = -1)

            next_index = torch.multinomial(probabilities, num_samples = 1)

            index = torch.cat((index, next_index), dim = 1)

            result = torch.cat((result, next_index), dim = 1)

            a, b = index.shape

            if b >= block_size:
                index = index[:, 1:]

        return result

# method to optimize and train the model

def train(model: GPTModel):

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for i in range (max_iterations):

        x, y = getBatch("train")

        logits, loss = model.forward(x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

    torch.save(model, model_path)

    print("saved")

# method to calculate the loss

@torch.no_grad()

def calculate_loss(model):

    model.eval()

    out={}

    splits = ['train', 'test']

    for split in splits:
        losses = torch.zeros(test_iterations)
        for k in range (test_iterations):

            x, y = getBatch(split)

            logits, loss = model.forward(x, y)

            losses[k] = loss.item()

        out[split] = losses.mean()
        model.train()
    return out

# defining the main method
    
def main():

    vocab_size = len(characters)

    model = GPTModel(vocab_size)

    # model = torch.load(model_path, weights_only= False)

    model = model.to(device)

    train(model)

    for i in range (max_test_iterations):

        loss = calculate_loss(model)
        if i % test_step_iterations == 9:
            
            print(f"At step {i+1} training loss: {loss['train']}, testing loss: {loss['test']}")

    while True:

        prompt = input("enter a prompt: ")
        context = torch.tensor(encode(prompt), dtype = torch.long, device=device)
        context = context.unsqueeze(0)

        generated_chars = decode(model.generate(context, max_sequence_length)[0].tolist())

        print(generated_chars)

if __name__ == "__main__":

    main()