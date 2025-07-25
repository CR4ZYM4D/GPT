import deepspeed
import torch
from model_architecture import GPTModel

model_path = './gpt/models/subset0'

model = GPTModel()

deepspeed_engine = deepspeed.init_inference(model = model, mp_size = 1, dtype = torch.float16, checkpoint = model_path)

inference_model = deepspeed_engine.module

def continueOrNot(prompt, inference_model, final_token_index):
    
    if final_token_index < inference_model.max_sequence_length-1:

        char = input("do you wish to continue further with this text? [y]es")

        if char.lower() == 'y':

            new_prompt = input("Enter continuation text: ")

            prompt += new_prompt

            result, final_token_index = inference_model.generate(prompt)

            if final_token_index < inference_model.max_sequence_length-1:
                print(result)
                continueOrNot(result, inference_model, final_token_index)

            else: 
                print("Final result: ", result)

        else:
            return
        
    else: 
        return

model_path = input("Enter model path relative to root folder: ")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = torch.load(model_path)

model = model.to(device)

while(True):

    prompt = input("Enter some prompt, exit to escape: ")

    if prompt.lower() == "exit":
        break

    result, final_token_index = model.generate(prompt)

    print(result)

    continueOrNot(result, model, final_token_index)


