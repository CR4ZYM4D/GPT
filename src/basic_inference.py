import torch

def continueOrNot(prompt, model, final_token_index):
    
    if final_token_index < model.max_sequence_length-1:

        char = input("do you wish to continue further with this text? [y]es")

        if char.lower() == 'y':

            new_prompt = input("Enter continuation text: ")

            prompt += new_prompt

            result, final_token_index = model.generate(prompt)

            if final_token_index < model.max_sequence_length-1:
                print(result)
                continueOrNot(result, model, final_token_index)

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
