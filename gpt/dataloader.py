import re
import os
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


# Read Training Data

def clean_text(text):
    # Modified to accept text directly instead of filename
    cleaned_text = re.sub(r'\n+', ' ', text) # 줄바꿈을 빈칸으로 변경
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text) # 여러 빈칸을 하나의 빈칸으로
    return cleaned_text

class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)
        print("# of tokens in txt:", len(token_ids))

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# Initialize tokenizer
tokenizer = tiktoken.get_encoding("gpt2")

# Get all files in the data directory that start with "cleaned_"
data_dir = "data"
cleaned_files = []
for file in os.listdir(data_dir):
    if file.startswith("cleaned_"):
        file_path = os.path.join(data_dir, file)
        cleaned_files.append(file_path)

print(f"Found {len(cleaned_files)} cleaned files:")
for file in cleaned_files:
    print(f"- {file}")

# Process all cleaned files and combine them
all_text = ""
cleaned_files = cleaned_files[:1]
for file_path in cleaned_files:
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        cleaned = clean_text(text)
        print(f"{file_path}: {len(cleaned)} characters after cleaning")
        all_text += cleaned + " "  # Add a space between texts from different files

print(f"\nTotal characters in combined text: {len(all_text)}")

# Create dataset and dataloader
max_length = 32  # Context window size
stride = 4       # Step size for sliding window
batch_size = 128

dataset = MyDataset(all_text, max_length=max_length, stride=stride)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

print(f"\nDataset created with {len(dataset)} samples")
print(f"Number of batches: {len(train_loader)}")

# Example of accessing a batch
dataiter = iter(train_loader)
x, y = next(dataiter)
print("\nExample batch shapes:")
print(f"Input shape: {x.shape}")
print(f"Target shape: {y.shape}")

# Show an example of the first sequence in the batch
print("\nExample sequence:")
print("Input:", tokenizer.decode(x[0].tolist()))
print("Target:", tokenizer.decode(y[0].tolist()))
