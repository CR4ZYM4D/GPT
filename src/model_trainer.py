import torch
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from model_architecture import GPTModel 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "./gpt/models/model.pkl"
train_set_path = "./gpt/dataset/subset"

print(device)

model = torch.load(model_path) if os.path.exists(model_path) else GPTModel()

model = model.to(device)

total_avg_perplexity = []

total_avg_loss = []

for j in tqdm(range(21), desc = "subset number"):

	src_directory = f"{train_set_path}{j}"

	with open("./gpt/dataset/completed_subsets.txt", 'r') as f:

		if src_directory in f.read():

			continue 

	directory_files = os.listdir(src_directory)

	num_files = len(directory_files)
	
	subset_loss = 0.0

	subset_perplexity = 0.0

	subset_avg_loss = []

	subset_avg_perplexity = []

	trained_files = set()

	if(os.path.exists(f"./gpt/dataset/subset{j}_trained_files.txt")):

		with open(f"./gpt/dataset/subset{j}_trained_files.txt", 'r')as f:

			trained_files = set(f.read())

	for i in tqdm(range(0, num_files, 8), leave = False):

		batch_files = directory_files[i: i+8]

		if batch_files not in trained_files:
	
			count= 0

			input_texts = []

			target_texts = []
		
			for file in batch_files:

				with open(file, 'r') as f:

					text = f.read()
				
					indices = torch.randint(low = len(text)//30, high = len(text)-1, size = (4,)).tolist()
				
					input_texts = [text[:index+1] for index in indices] if count == 0 else input_texts.append([text[:index+1] for index in indices])

					target_texts = [text[1: ]] if count == 0 else target_texts.append(text[1: ])

					count += 1

			logits, loss = model.train(input_texts, target_texts)

			torch.save(model, model_path)

			trained_files.update(batch_files)

			with open(f"./gpt/dataset/subset{j}_trained_files.txt", 'w')as f:

				f.write(f"{file}\n" for file in batch_files)

			subset_loss += loss.item()

			subset_perplexity += torch.exp(loss).item()

			subset_avg_loss.append(subset_loss/(i//8 + 1))
			
			subset_avg_perplexity.append(subset_perplexity / (i//8 +1))

	print(f"subset number {j} completed, total and average loss in subset is: {subset_loss: .3f} and {subset_avg_loss:.4f}")
	print(f"total and average perplexity in subset is: {subset_perplexity: .3f} and {subset_avg_perplexity: .4f}")

	sns.lineplot(y = subset_avg_loss, x = list(range(len(subset_avg_loss)+1)))
	plt.title(f"subset {j} average loss v/s number of batches trained")
	plt.xlabel("number of batches")
	plt.ylabel("average cross entropy loss")
	plt.savefig(f"./gpt/graphs/loss/subset loss/subset{j}.jpg")

	sns.lineplot(y = subset_avg_perplexity, x = list(range(len(subset_avg_perplexity)+1)))
	plt.title(f"subset {j} average perplexity v/s number of batches trained")
	plt.xlabel("number of batches")
	plt.ylabel("average perplexity")
	plt.savefig(f"./gpt/graphs/perplexity/subset perplexity/subset{j}.jpg")

	total_avg_loss.append(subset_avg_loss / (len(total_avg_loss)+1))

	total_avg_perplexity.append(subset_avg_perplexity / len(total_avg_perplexity)+1)

	with open("./gpt/dataset/completed_subsets.txt", 'w') as f:

		f.write(f"{src_directory}\n")

sns.lineplot(y = total_avg_loss, x = list(range(1,22)))
plt.title("model average loss v/s number of subsets trained")
plt.xlabel("number of subsets")
plt.ylabel("average cross entropy loss")
plt.savefig("./gpt/graphs/loss/model.jpg")

sns.lineplot(y = total_avg_perplexity, x = list(range(1,22)))
plt.title("model average perplexity v/s number of subsets trained")
plt.xlabel("number of subsets")
plt.ylabel("average perplexity")
plt.savefig("./gpt/graphs/perplexity/model.jpg")
