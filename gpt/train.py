import torch
import os
import matplotlib.pyplot as plt
from dataloader import MyDataset, clean_text, tokenizer
from nnet import GPTModel
from torch.utils.data import DataLoader
import time
from datetime import datetime, timedelta

def plot_losses(losses, save_path='training_loss.png'):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.savefig(save_path)
    plt.close()

def train_model(
    batch_size=128,
    max_length=32,
    stride=4,
    learning_rate=0.0004,
    weight_decay=0.1,
    num_epochs=100,
    save_interval=1,
    device=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Initial GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

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
    for file_path in cleaned_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            cleaned = clean_text(text)
            print(f"{file_path}: {len(cleaned)} characters after cleaning")
            all_text += cleaned + " "

    print(f"\nTotal characters in combined text: {len(all_text)}")

    # Create dataset and dataloader
    dataset = MyDataset(all_text, max_length=max_length, stride=stride)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Number of batches: {len(train_loader)}")

    # Initialize model and optimizer
    torch.manual_seed(123)
    model = GPTModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training loop
    tokens_seen = 0
    global_step = -1
    losses = []
    
    start_time = time.time()
    last_print_time = start_time
    
    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        epoch_loss = 0
        batch_times = []
        
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            batch_start = time.time()
            
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            
            tokens_seen += input_batch.numel()
            global_step += 1
            
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            # Print progress every 5 seconds or every 10 batches
            current_time = time.time()
            if current_time - last_print_time > 5 or (batch_idx + 1) % 10 == 0:
                avg_batch_time = sum(batch_times[-10:]) / len(batch_times[-10:])
                batches_remaining = len(train_loader) - (batch_idx + 1)
                est_epoch_time = batches_remaining * avg_batch_time
                
                print(f"\rBatch {batch_idx + 1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Tokens/sec: {input_batch.numel()/batch_time:.1f} | "
                      f"Batch time: {batch_time:.2f}s | "
                      f"Est. remaining: {timedelta(seconds=int(est_epoch_time))}", end="")
                
                if torch.cuda.is_available():
                    print(f" | GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB", end="")
                
                last_print_time = current_time
        
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        print(f"\n\nEpoch {epoch + 1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Epoch Time: {timedelta(seconds=int(epoch_time))}")
        print(f"Average Batch Time: {sum(batch_times)/len(batch_times):.2f}s")
        print(f"Total Tokens Seen: {tokens_seen}")
        
        if torch.cuda.is_available():
            print(f"GPU Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB")

        # Save model checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join('checkpoints', f"model_{epoch + 1:03d}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    total_time = time.time() - start_time
    print(f"\nTraining completed in {timedelta(seconds=int(total_time))}")
    
    # Plot and save the training loss curve
    plot_losses(losses)
    
    # Save final model
    final_model_path = os.path.join('checkpoints', "model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == "__main__":
    train_model()