from transformers import AutoTokenizer

from tokenizers.processors import TemplateProcessing

# importing tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")


special_tokens_to_add = {'bos_token' : "<|startoftext|>", "pad_token": "<|padding|>", "unk_token": "<|unknown|>"}

tokenizer.add_special_tokens(special_tokens_to_add)

tokenizer._tokenizer.post_processor = TemplateProcessing(
    single="<|bos|> $A <|eos|>",
    pair="<|bos|> $A <|eos|> $B:1 <|eos|>:1",
    special_tokens=[
        ("<|bos|>", tokenizer.bos_token_id),
        ("<|eos|>", tokenizer.eos_token_id),
    ],
)

tokenizer_local_path = "./gpt/models/tokenizer"

tokenizer.save_pretrained(tokenizer_local_path)

print(len(tokenizer))