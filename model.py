import torch
import torch.nn as nn
import torch.nn.functional as F

# important constants to be used in the model

block_size = 9 # size of a single word or a combination of words (we will refer to this a s a block)

batch_size = 12 # no. of said blocks or words that we will handle at once

vector_dimension = 512 # dimensions of each of the alphabet or token vector\

max_sequence_length = 400 # max no of tokens that will be generated

learning_rate = 3e-2

max_iterations = 200

train_step_iteration = 10

max_test_iterations = 200

test_iterations = 20

test_step_iterations = 10

# using GPU if available

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# function to read a file and return the set of all characters in it for tokenization

def readTextFile(path: str):

    with open(path, 'r', encoding = 'UTF-8') as f:

        text = f.read()

    characters = sorted(set(text))

    return text, characters

text, characters = readTextFile('./src/wizardOfOz.txt')

#lambda functions to encode and decode  the text

stringToNum = {ch: i for i, ch  in enumerate(characters)}

numToString = {i: ch for i,ch in enumerate(characters)}

encode = lambda s: [stringToNum[c] for c in s]

decode = lambda s: ''.join(numToString[n] for n in s)

#converting the text into a tensor

data = torch.tensor(encode(text), dtype = torch.long)

# setting training and testing set sizes as 80% and 20%

training_set_size = int( len(data) * 0.8)

testing_set_size = len(data) - training_set_size

training_set = data[:training_set_size]
testing_set = data[training_set_size:]

# function to split and return the training and testing batches

def getBatch(split = "train"):
    
    index = torch.randint(high = training_set_size - block_size -1 if split == "train" else testing_set_size-block_size-1, size=(batch_size,))

    if split == "train":

        x = torch.stack([training_set[ix: ix + block_size] for ix in index])

        y = torch.stack([training_set[ix + 1: ix + block_size + 1] for ix in index])

        x, y = x.to(device), y.to(device)

    else:

        x = torch.stack([testing_set[ix: ix + block_size] for ix in index])

        y = torch.stack([testing_set[ix + 1: ix + block_size + 1] for ix in index])

        x, y = x.to(device), y.to(device)

    return x, y

class GPTModel(nn.Module):

    # constructor

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        self.token_embeddings = nn.Embedding(vocab_size, vocab_size)
    
    # method to return the token embedding of given index token and return the loss if there is a target set given

    def forward(self, index, targets= None):

        # getting the token embedding of the given token index
 
        logits = self.token_embeddings(index)
        
        if targets == None:

            loss=None

        else: 

            batch, time, channels = logits.shape

            # making logits and targets of similar dimensions to calculate the loss

            logits = logits.view(batch * time, channels)

            targets = targets.view(batch * time)

            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    # method to generate the tokens

    def generate(self, index, max_sequence_length):
    
        for _ in range (max_sequence_length):

            logits, loss = self.forward(index)

            logits = logits[: , -1, :]

            probabilities = F.softmax(logits, dim = -1)

            next_index = torch.multinomial(probabilities, num_samples = 1)

            index = torch.cat((index, next_index), dim = 1)

        return index

# method to optimize and train the model

def train(model: GPTModel):

    optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

    for i in range (max_iterations):

        x, y = getBatch("train")

        logits, loss = model.forward(x, y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        if i % train_step_iteration == 0:

            print(loss.item())

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

    model = model.to(device)

    train(model)

    for i in range (max_test_iterations):

        loss = calculate_loss(model)
        if i % test_step_iterations == 9:
            
            print(f"At step {i+1} training loss: {loss['train']}, testing loss: {loss['test']}")

    context = torch.zeros((1,1), dtype = torch.long, device=device)
    generated_chars = decode(model.generate(context, max_sequence_length)[0].tolist())

    print(generated_chars)

if __name__ == "__main__":

    main()