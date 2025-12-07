"""
Test script to load saved models and run text generation.
"""

import torch
import torch.nn.functional as F
from transformer import Transformer
from moes import RegularMoE, RandomMoE, OrthogonalMoE, HashMoE, ShittyMoE, NonLinearMoE
from tokenizers import Tokenizer
import glob
import os

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Load tokenizer (must match training)
print("Loading tokenizer...")
# Try multiple locations
if os.path.exists("tokenizer.json"):
    tokenizer_obj = Tokenizer.from_file("tokenizer.json")
elif os.path.exists("code/tokenizer.json"):
    tokenizer_obj = Tokenizer.from_file("code/tokenizer.json")
else:
    raise FileNotFoundError("tokenizer.json not found in current directory or code/")

class SimpleTokenizer:
    def __init__(self, tok):
        self.tok = tok
        self.pad_token_id = 0
        
    def encode(self, text):
        return self.tok.encode(text).ids
    
    def decode(self, ids):
        # Convert to list if tensor
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        # Filter out padding tokens
        ids = [i for i in ids if i != self.pad_token_id]
        # Decode using tokenizer's built-in decoder (handles ByteLevel properly)
        return self.tok.decode(ids)

tokenizer = SimpleTokenizer(tokenizer_obj)

# Mapping from model names to MoE classes
MOE_CLASSES = {
    'RegularMoE': RegularMoE,
    'RandomMoE': RandomMoE,
    'OrthogonalMoE': OrthogonalMoE,
    'HashMoE': HashMoE,
    'ShittyMoE': ShittyMoE,
    'NonLinearMoE': NonLinearMoE,
}

def load_model(checkpoint_path):
    """Load model from checkpoint"""
    print(f"\nLoading {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model_name = checkpoint['model_name']
    
    # Get the appropriate MoE class
    moe_class = MOE_CLASSES[model_name]
    moe_fn = lambda: moe_class(config['D'], config['H'], config['N'], config['K'])
    
    # Reconstruct model
    model = Transformer(
        config['V'], config['D'], config['n_heads'], config['n_layers'],
        moe_fn, config['max_seq_len']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model: {model_name}")
    print(f"  Train loss: {checkpoint['final_train_loss']:.4f}")
    print(f"  Test loss: {checkpoint['final_test_loss']:.4f}")
    
    return model, model_name, config

def generate(model, prompt, max_tokens=50, temperature=1.0, top_k=50):
    """Generate text continuation given a prompt"""
    # Encode prompt
    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)  # [1, L]
    
    generated = tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            # Get current sequence (keep within max_seq_len)
            curr_seq = generated[:, -model.blocks[0].attn.causal_mask.size(0):]
            
            # Forward pass
            logits = model(curr_seq, mask_BS=None)  # [1, L, V]
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature  # [1, V]
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                next_token_logits.scatter_(1, top_k_indices, top_k_logits)
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if we hit padding token (optional)
            if next_token.item() == tokenizer.pad_token_id:
                break
    
    return tokenizer.decode(generated.squeeze(0).cpu().numpy())

# Load all saved models
print("="*60)
print("LOADING MODELS")
print("="*60)

# Try to find saved models in multiple locations
model_paths = glob.glob("saved_models/model_*.pth")
if not model_paths:
    model_paths = glob.glob("code/saved_models/model_*.pth")
if not model_paths:
    print("No saved models found in saved_models/ or code/saved_models/")
    print("Please run exp.py first to train and save models")
    exit(1)

models = []
for path in sorted(model_paths):
    model, name, config = load_model(path)
    models.append((model, name))

# Test generation with multiple prompts
print("\n" + "="*60)
print("TEXT GENERATION")
print("="*60)

prompts = [
    "The stock market",
    "Scientists discovered",
    "In a major development",
]

for prompt in prompts:
    print(f"\n{'='*60}")
    print(f"PROMPT: '{prompt}'")
    print(f"{'='*60}")
    
    for model, name in models:
        print(f"\n{name}:")
        generated = generate(model, prompt, max_tokens=30, temperature=0.8, top_k=50)
        print(f"  {generated}")

print("\n" + "="*60)
print("DONE")
print("="*60)

