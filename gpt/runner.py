import torch
from dataloader import tokenizer
from nnet import GPTModel, CONTEXT_LENGTH

def generate(model, prompt, max_new_tokens=50, temperature=0.8, top_k=40, device=None):
    """
    Generate text from the model given a prompt.
    
    Args:
        model: The trained GPT model
        prompt: String prompt to start generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Controls randomness (0.0 = deterministic, 1.0 = most random)
        top_k: Number of highest probability tokens to consider for sampling
        device: Device to run generation on (defaults to cuda if available)
    
    Returns:
        Generated text string
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    
    # Encode the prompt
    idx = tokenizer.encode(prompt)
    idx = torch.tensor(idx).unsqueeze(0).to(device)
    
    # Generate tokens
    for _ in range(max_new_tokens):
        # Take last context_size tokens as input
        idx_cond = idx[:, -CONTEXT_LENGTH:]
        
        # Get model predictions
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Apply top-k filtering
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)
        
        # Apply temperature and sample
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        # Append the new token
        idx = torch.cat((idx, idx_next), dim=1)
        
        # Check if we generated a newline or period (simple stopping condition)
        if idx_next.item() in [tokenizer.encode("\n")[0], tokenizer.encode(".")[0]]:
            break
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(idx[0].tolist())
    return generated_text

def load_model(checkpoint_path, device=None):
    """Load a trained model from a checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = GPTModel()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    # Load the trained model
    model = load_model("checkpoints/model_002.pth")
    
    print("GPT Model loaded and ready for text generation!")
    print("Enter 'quit' to exit")
    
    while True:
        prompt = input("\nEnter a prompt: ")
        if prompt.lower() == 'quit':
            break
        
        # Generate text
        try:
            generated_text = generate(model, prompt)
            print("\nGenerated text:")
            print(generated_text)
        except Exception as e:
            print(f"Error during generation: {e}")

if __name__ == "__main__":
    main() 