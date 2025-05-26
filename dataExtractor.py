import os
import lzma
from tqdm import tqdm

wsl_dataset_path = "/mnt/c/Users/Madhav/Github Repos/gpt/dataset/openwebtext/"

windows_dataset_path = "c:/Users/Madhav/Github Repos/gpt/dataset/openwebtext/"

vocabulary_file = "./dataset/vocab.txt"

vocabulary = set()

training_data = "./dataset/training data.txt"

testing_data = "./dataset/testing data.txt"

def getFiles(directory: str):

    files=[]

    for file in os.listdir(directory):

        if file.endswith(".xz") and os.path.isfile(os.path.join(directory, file)):

            files.append(file)

    return files

def extractFiles(split_file, result_file_name, folder_path):

    with open(result_file_name, "w", encoding = "utf-8") as result_dataset:

        for filename in tqdm(split_file, total = len(split_file)):

            file_path = os.path.join(folder_path, filename)

            with lzma.open(file_path, "rt", encoding = "utf-8") as in_data:

                text = in_data.read()

                result_dataset.write(text)

                characters = set(text)

                vocabulary.update(characters)

files = getFiles(wsl_dataset_path)

split_index = int(0.8 * len(files))

train_files = files[:split_index]

test_files = files[split_index:]

extractFiles(train_files, training_data, wsl_dataset_path)

extractFiles(test_files, testing_data, wsl_dataset_path)

with open(vocabulary_file, "w", encoding = "utf-8") as file:

    for char in vocabulary:
        file.write(char + "\n")

