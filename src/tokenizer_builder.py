from transformers import AutoTokenizer

# importing tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")


special_tokens_to_add = {'bos_token' : "<|startoftext|>", "pad_token": "<|padding|>", "unk_token": "<|unknown|>"}

tokenizer.add_special_tokens(special_tokens_to_add)

tokenizer_local_path = "./gpt/models/tokenizer"

tokenizer.save_pretrained(tokenizer_local_path)

print(len(tokenizer))