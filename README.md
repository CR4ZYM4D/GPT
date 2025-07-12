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

These matrices from each attention head are returned and concatenated back along the head dimesnion to regain our matrix of _sequence length X embedding dimension_. This tensor is then multiplied with another matrix of _embedding dimension X embedding dimension_ called the _soutput weights_ and we finally obtain the contextualized embeddings.


Note: If there is any code in the root directory, **_do not_** refer to it as it will be dropped soon
