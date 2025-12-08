# Same as the exp.ipynb file, but in a python script

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import einops
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborne as sns
import time

from transformer import Transformer
from moes import RegularMoE, RandomMoE, OrthogonalMoE, HashMoE, UniformMoE, NonLinearMoE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu" and torch.backends.mps.is_available():
    device = torch.device("mps")
    torch.mps.manual_seed(67960)
if device.type == "cuda" or device.type == "cpu":
    torch.manual_seed(67960)

B = 128
V = 1024

print(f"Using device: {device}")


from datasets import load_dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders
import torch

# load dataset
print("Loading AG News dataset...")
dataset = load_dataset("ag_news")
train_data = dataset['train']
test_data = dataset['test']

# Train a simple BPE tokenizer on AG News (vocab V)
print("Training BPE tokenizer with vocab_size=V...")
tokenizer_obj = Tokenizer(models.BPE())
tokenizer_obj.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer_obj.decoder = decoders.ByteLevel()

# Train on first 10k samples (fast: ~10 seconds)
trainer = trainers.BpeTrainer(vocab_size=V, special_tokens=["<PAD>", "<UNK>"])
tokenizer_obj.train_from_iterator(
    (train_data[i]['text'] for i in range(min(10000, len(train_data)))),
    trainer=trainer
)

# Add post-processor to handle byte-level decoding properly
tokenizer_obj.post_processor = processors.ByteLevel(trim_offsets=False)

# Save tokenizer for later use
tokenizer_obj.save("tokenizer.json")
print("Saved tokenizer to tokenizer.json")

# Simple wrapper to match expected API
class SimpleTokenizer:
    def __init__(self, tok):
        self.tok = tok
        self.pad_token_id = 0
        
    def __call__(self, text, truncation=True, max_length=128, padding='max_length', return_tensors='pt'):
        encoding = self.tok.encode(text)
        tokens = encoding.ids[:max_length]
        tokens = tokens + [self.pad_token_id] * (max_length - len(tokens))
        return {'input_ids': torch.tensor([tokens])}
    
    def decode(self, ids):
        return self.tok.decode(ids.tolist() if isinstance(ids, torch.Tensor) else ids)

tokenizer = SimpleTokenizer(tokenizer_obj)
vocab_size = tokenizer_obj.get_vocab_size()

print(f"Vocab size: {vocab_size}")
print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}")

# dataset class
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, max_len=128):
        self.data = hf_dataset
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]['text']
        encoding = tokenizer(text, max_length=self.max_len+1)
        tokens = encoding['input_ids'].squeeze(0)
        
        x = tokens[:-1]
        y = tokens[1:]
        mask = (x != tokenizer.pad_token_id)
        return x, y, mask

# create dataloaders
train_dataset = TextDataset(train_data, max_len=128)
test_dataset = TextDataset(test_data, max_len=128)
train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=B, shuffle=False, num_workers=0)

print("\n*Data done loading*")


D = 256
H = 512
N = 32
K = 4
V = vocab_size
n_heads = 8
n_layers = 6
max_seq_len = 128

print(f"D: {D}\n H: {H}\n N: {N}\n K: {K}\n V: {V}\n n_heads: {n_heads}\n n_layers: {n_layers}\n max_seq_len: {max_seq_len}")

# create models
moe_fns = [
    lambda: RegularMoE(D, H, N, K),
    lambda: RandomMoE(D, H, N, K),
    lambda: OrthogonalMoE(D, H, N, K),
    lambda: HashMoE(D, H, N, K),
    lambda: UniformMoE(D, H, N, K),
    lambda: NonLinearMoE(D, H, N, K)
]
models = [Transformer(V, D, n_heads, n_layers, moe_fn, max_seq_len).to(device) for moe_fn in moe_fns]
model_names = [moe_fns[i]().__class__.__name__ for i in range(len(moe_fns))]

# Convert to BFloat16 on CUDA for mixed precision (momoe requires this)
# if device.type == "cuda":
#     models = [model.to(torch.bfloat16) for model in models]

# print number of parameters in each model
for i, model in enumerate(models):
    print(f"Model {i+1} ({model_names[i]}) has {sum(p.numel() for p in model.parameters())} parameters and {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters")


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch, return average loss"""
    model.train()
    total_loss = 0
    num_batches = 0

    print(f"{len(loader)} batches to process...")
    
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        
        # tm1 = time.time()
        logits = model(x, mask)  # [B, S, V]
        # tm2 = time.time()
        # print(f"Time taken for forward pass: {tm2 - tm1:.4f}s")
        
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=tokenizer.pad_token_id)
        
        # tm1 = time.time()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # tm2 = time.time()
        # print(f"Time taken for backward pass: {tm2 - tm1:.4f}s")
        
        total_loss += loss.item()
        num_batches += 1
        if num_batches % 100 == 0:
            print(f"Processed {num_batches} batches...")
    return total_loss / num_batches

@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate on dataset, return average loss"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    for x, y, mask in loader:
        x, y, mask = x.to(device), y.to(device), mask.to(device)
        logits = model(x, mask)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1), ignore_index=tokenizer.pad_token_id)
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

# Training config
num_epochs = 3
lr = 3e-4

# Train each model
results = {}
for i, (model, name) in enumerate(zip(models, model_names)):
    print(f"\n{'='*60}")
    print(f"Training Model {i+1}/{len(models)}: {name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    # if device.type == "cuda":
    #     model = model.to(torch.bfloat16)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate
        test_loss = evaluate(model, test_loader, device)
        test_losses.append(test_loss)
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Time: {epoch_time:.2f}s")
    
    results[name] = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1]
    }
    
    # Move back to CPU to free memory
    model = model.cpu()

# Print summary
print(f"\n{'='*60}")
print("FINAL RESULTS")
print(f"{'='*60}")
for name, res in results.items():
    print(f"{name:20s} | Train: {res['final_train_loss']:.4f} | Test: {res['final_test_loss']:.4f}")

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, res in results.items():
    ax1.plot(range(1, num_epochs+1), res['train_losses'], marker='o', label=name)
    ax2.plot(range(1, num_epochs+1), res['test_losses'], marker='o', label=name)

ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Test Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# save the plot results
plt.savefig("plots/loss_plot.png")



for i in range(len(models)):
    model = models[i]
    for j, blk in enumerate(model.blocks):
        print(f"Model {i+1} ({model_names[i]}), Layer {j+1}: {blk.moe.biases_N}")




# Test perplexity on a single example
test_idx = 0
test_text = test_data[test_idx]['text']
print(f"Test example {test_idx}:")
print(f"Text: {test_text[:200]}...")
print()

# Tokenize
encoding = tokenizer(test_text, truncation=True, max_length=129, 
                     padding='max_length', return_tensors='pt')
tokens = encoding['input_ids'].squeeze(0)
x = tokens[:-1].unsqueeze(0).to(device)  # [1, 128]
y = tokens[1:].unsqueeze(0).to(device)   # [1, 128]
mask = (x != tokenizer.pad_token_id)

# Save each model to its own place
import os
os.makedirs("saved_models", exist_ok=True)

for i, model in enumerate(models):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_name': model_names[i],
        'config': {
            'V': V, 'D': D, 'H': H, 'N': N, 'K': K,
            'n_heads': n_heads, 'n_layers': n_layers,
            'max_seq_len': max_seq_len
        },
        'final_train_loss': results[model_names[i]]['final_train_loss'],
        'final_test_loss': results[model_names[i]]['final_test_loss'],
    }
    torch.save(checkpoint, f"saved_models/model_{model_names[i]}.pth")
    print(f"Saved {model_names[i]} to saved_models/model_{model_names[i]}.pth")



# # Test each model
# print("="*60)
# print("PERPLEXITY RESULTS")
# print("="*60)

# for model, name in zip(models, model_names):
#     model = model.to(device)
#     # if device.type == "cuda":
#     #     model = model.to(torch.bfloat16)
#     model.eval()
    
#     with torch.no_grad():
#         logits = model(x, mask)  # [1, S, V]
#         loss = F.cross_entropy(logits.view(-1, V), y.view(-1), 
#                                ignore_index=tokenizer.pad_token_id, reduction='mean')
#         perplexity = torch.exp(loss).item()
    
#     print(f"{name:20s} | Loss: {loss.item():.4f} | Perplexity: {perplexity:.2f}")
#     model = model.cpu()

# print()
# print("Lower perplexity = better prediction")
# print("(Perplexity measures how 'surprised' the model is by the actual next token)")