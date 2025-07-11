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

## Model Components And Their Purpose

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

These tokens along with the |endoftext| token help the model understand what part of the token "list" does it need to consider,
adding filler i.e the padding token to make sure all prompts have the same length and an unknown token to add when the model does not know what that character / substring is (if using non english langugaes mainly)

### 2. Token Embeddings:

#### Purpose






Note: If there is any code in the root directory, **_do not_** refer to it as it will be dropped soon
