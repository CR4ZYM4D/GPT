# Language Models Are Unsupervised Multi-task learners 

This is an implementation of the 2019 Research Paper [Language Models Are Unsupervised Multi-Task Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) published by OpenAI which describes the GPT-2 Architecture.

**The Readme.md file explains the flow of data, datasets used, various components and their purpose and the training procedure**

## What does it aim to achieve?

This Repository has been made as an attempt to help people understand how at a crude level, current [autoregressive](https://www.google.com/search?client=firefox-b-e&channel=entpr&q=autoregressive+models) AI models work i.e by implementing very basic LLM trained on an extremely large dataset, followed by fine-tuning them and lastly, using reinforcement learning with human feedback to teach the model what to speak.

_There is right now no plan to cover the latter 2 of these 3 training steps._ 

## Dataset Used

The dataset used to train the LLM is the [OpenWebText dataset](https://huggingface.co/datasets/Skylion007/openwebtext/tree/main), an open source version of the WebText dataset used by OpenAI to train their GPT-2 model.
Both of these datasets are compilation of some of the most commented reddit threads on the website from 2005 - 2019 which were then preprocessed, refined and stored as millions of .txt files. Both datasets having a size of ~40GB each.

## Model Architecture

The model uses a modified decoder block architecture of the 2017 transformer model paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762) by Google. **The major differences being:**

**1. The transformer using both an encoder and decoder while GPT-2 using only the decoder block**

**2. The transformer using both a self and cross Multi-Head Attention Mechanism while GPT-2 using only the self Multi-Head Attention due to the obvious lack of an encoder**



<img width="177" height="383" alt="image" src="https://github.com/user-attachments/assets/473265a3-10a1-4513-afd8-40892e1991d8" /> <img width="292" height="400" alt="image" src="https://github.com/user-attachments/assets/ddd46781-ce74-46f6-8cf8-d850e9f333ef" />

Figure. The GPT-2 (left) and Transformer (right) Architectures

## Model Components, Data Flow And Their Purpose

### 1. Tokenizer: 

#### What is it's purpose?

Current AI models **_DO NOT_** understand language like we humans do. Like any other computer program or software to them it is just a gibberish collection of characters. What they do understand is numbers, and to get them to function as per our needs we need to find a way to convert these strings into a set of numbers and that is exactly what a tokenizer does.


A tokenizer takes a string or prompt or text as input and converts it into a list like collection of numbers. 

Tokenizers are mainly of 3 types: 

_Word Level, Character Level and Sub-Word Level._

There are also various methods of tokenization such as bag of words or one hot encoding but the one mainly used in Large Language Models is [Byte Pair Encoding](https://www.google.com/search?client=firefox-b-e&channel=entpr&q=byte+pair+encoding)

The tokenizer converts the string into a list or tensor of token indices which are then sent to the language model.

In this project we use the pre-built GPT-2 tokenizer via hugging face but if you want to know how to build your own, [this would be a good starting point](https://www.youtube.com/watch?v=zduSFxRajkE&t=2204s&pp=ygUdbGV0J3MgYnVpbGQgdGhlIGdwdCB0b2tlbml6ZXI%3D)

The code on how to import and use the tokenizer is in the tokenizer_builder.py file and is pretty self explanatory.

_Do note that the default tokenizer has no |startoftext|, |padding| or |unknown| tokens which have been added manually in the file as well._

These tokens along with the |endoftext| token help the model understand what part of the token "list" / tensor does it need to consider,
adding filler i.e the padding token to make sure all prompts have the same length and an unknown token to add when the model does not know what that character / substring is (if using non english langugaes mainly)

### 2. Token Embeddings:

#### Purpose

The problem that an LLM faces is that with almost very language, one word can have more than one meanings. Furthermore, to make it not spout nonsense, it needs to capture some essence of the words that are given to it. Both of these problems cannot be solved by just giving the LLM just a single number representing the index value of the token. 

The solution? 

Give it an entire tensor of numbers, unique for each token index called it's token embedding. 

This token embedding is essentially a multi dimesnional vector that contains floating point numbers which help represent the various meaning of that token to the model. The dimensions of this vector can be determined by the programmer and are referred to as _"embedding dimension"_. 

A higher embedding dimension while means that the model will understand various words and tokens better, it also requires a higher compute power.

An Embedding Table can be though of as a Relational Database or a Hash Map, where you provide the token indexes as keys and it returns the embedding vector as values. The token list obtained from the tokenizer is then passed into the token embeddings to obtain the token tensors and turns into a 2-Dimesnional Matrix of dimensions 

_length of prompt X the embedding dimension_

### 3. Positional Encodings: 

#### Purpose

Similar to how a word can have multiple meanings, the positioning of particular words or phrases in sentences can also more often than not, completely alter the meaning that the sentence implies. 

_For example, "he only loves dogs" and "only he loves dogs" while having the same words, mean completely different_

To counter this, similar to the token embeddings we also make an embedding for the positions of tokens in a prompt. **_Note that all these embeddings should have the same embedding dimension_**

We add simply add these positional encodings or embeddings to the token embeddings, pass them through a [**Layer Normalization layer**](https://www.geeksforgeeks.org/deep-learning/what-is-layer-normalization/) and now our prompt is finally ready to be used by the LLM 

### 4. Decoder Block:

This is the main block of the model which contains the attention mechanism. Multiple of these blocks are stacked together on top of each other with the output of one being the input to the next forming the basis of thse LLM models. 

The number of these blocks to be stacked / layers to be used is also a hyper-parameter (parameter defined by programmer) but generally 8 is good spot.

#### i. Layer Normalization:

The first component present in a decoder block is a layer normalization layer to normalize the input tensors which are the _token embeddings + positional encodings_ for the first decoder block and the output of the previous block for all of the following blocks

A copy of the normalized tensor is also made for later use.

#### ii. Masked Multi-Head Self Attention:

This block is how any LLM like GPT, Gemini, Llama or Claude are able to understand what to focus on or _attend to_ and what to ignore?   

The normalized token embeddings are passed into the attention blocks where 3 more of its copies are made.

The three copies are multiplied with three square matrices namely _key, query and value weights_, one with each yielding _key, query and value results_ respectively. All of the matrices having dimensions _embedding_dimension X embedding_dimension_.

The query results matrix is then multiplied with the transpose of the key results matrix to yield a matrix of dimensions _sequence length X sequence length_. This matrix aims to present how well different parts of thesequence relate to each other or to represnt the dot product between the tensors of the matrix. A higher value representing a stronger inter-connection between the two tokens. _After multiplying, we also divide the resulting matrix with the square root of the enmbedding dimension mainly to ensure that any tensor element value does not become too high that it solely starts influencing the model_

These result matrices are now split along the embedding dimension into multiple chunks. A chunk of each of the result and product matrices is passed to a separate attention head. The dimensions of the embedding received by each chunk is equal and called the head dimension. This is done with the intention that each attenion head learns about a different way that the word is used within the sentence such as verb or adjective or noun and calculate its attention w.r.t. that specific way within the sequence    

Now the causal mask is applied to the this product. The causal mask replaces every element above the major diagonal of the matrix to minus infinity. **_This needs to be done only for Language Models as we need to predict the next token and thus it can not be taken into consideration in the attention mechanism. In vision transformers this causal mask need not be applied_**

Now we take the softmax of this updated matrix, the minus infinity term becomes a zero resulting in a matrix where every element above the major diagonal is zero and the lower elements present a fractional probability like score of how deeply inter-connected the two tokens constituting that product are with each other.

Finally this matrix is multiplied with the query results matrix to once agin obtain a matrix of dimensions _sequence length X head dimension_

These matrices from each attention head are returned and concatenated back along the head dimesnion to regain our matrix of _sequence length X embedding dimension_. This tensor is then multiplied with another matrix of _embedding dimension X embedding dimension_ called the _output weights_ and we finally obtain the contextualized embeddings.

#### iii. Layer Normalization: 

The output of the multi head attention block is then added with the normalized token embeddings that were originally sent to the Multi-Head Attention Block by what is caleed a residual connection and passed through another layer of Layer Normalization

#### iv. Feed Forward Network: 

The twice normalized result embeddings are then passed through a simple Feed-Forward Neural Network(FFN) or Multi-Layer Perceptron(MLP). This is usually done to help remove the linearity in the model and thus a s a result help it generalize better. The normalized-contextualized embeddings are passed through a linear layer which scales the embedding dimension to generally(but not necessarily so) four times it's original size. These are then passed through some activation functions (GeLU is mostly used with LLMs because it gives better results but ReLU can also be used, it was found in research that it makes training LLMs hard and is thus avoided) and then scaled back to the original embedding dimension and sent forward to the next block.

#### v. The Next Block: 

The result of the normalized MLP outputs are then added to the normalized non-MLP outputs through another residual connection and passed on to the next decoder block, _if there are any more decoder blocks left_  or to the embedding layer for the final steps.

### 5. Embedding Layer:

The fully processed, normalized, contextualized and every other fancy word-ified tensors of the original input sequence are passed into a second embedding layer to convert the tensors back from that of size _batch size x sequence length x embedding dimension_ to those of size _batch size x sequence length x vocabulary size_. These tensors are then passed along to the final layer.

### 6. Softmax Layer:

The softmax fucntion is applied on the tensors of dimensions _batch size x sequence length x vocabulary size_ along the last dimension i.e the vocabulary size. This gives us a probablistic distribution of what the model thinks the next token should be after this current one. For parts of the sequence it should be the same as the next token provided in the sequence itself. Then we get the token that the model thinks should come after the last token of the sequence, which is either the one with the highest softmax value or some other token depending on the degree of randomness of the model called as *_temperature_*. This one single token is appended to the original sequence and the process is repeated with the modified sequence until the model thinks that it should be complete now (by returning the *|endoftext|* token ) or our model runs out of available sequence length to continue.

## Training

The training loop in itself is pretty trivial, we iterate through each of the 21 subsets one at a time, select 8 text files for a batch and read their text completely. The text from the second token is what we wish to expect the model to return. We also select 4 random indices from each of the 8 files making the effective batch size as 32, and send the text until those indices as the input_tokens and the text from the second token onwards as the desired result. Cross-Entropy loss is calculated against the two and trained the model upon. Scaler and autocast are mainly used for optimizing vRAM usage, while tensorboard is used for the loss and perplexity curve plotting and profiler for the tracking of GPU and CPU usage.

## Usage 

Just install the required libraries, load the model and call the generate method.


Note: If there is any code in the root directory, **_do not_** refer to it as it will be dropped soon

_GPU compute is expensive, while everything has been proof checked and verified to work perfectly fine on my local GPU, it simply does not have the vRAM or power to actually train it (furthermore it takes around 150 hrs to train the entire model on my GPU with an embedding dimension of 256 and sequence length of 128). If any model is not available, then I am currently arranging the funds to train it using cloud platforms._
