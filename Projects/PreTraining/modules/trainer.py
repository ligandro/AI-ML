"""
Training utilities for transformer pretraining
"""

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm


def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits, _ = model(input_ids)
        
        # Calculate loss
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)


def evaluate(model, val_loader, criterion, device):
    """Evaluate on validation set"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            logits, _ = model(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity


def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=50, device='cpu'):
    """Generate text from a prompt using top-k sampling"""
    model.eval()
    
    # Encode prompt
    tokens = [tokenizer.word2idx.get('<BOS>')] + tokenizer.encode(prompt)
    
    with torch.no_grad():
        for _ in range(max_length):
            # Prepare input
            input_ids = torch.tensor([tokens]).to(device)
            
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get logits for next token
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probs = F.softmax(top_k_logits, dim=-1)
            
            # Sample next token
            next_token_idx = torch.multinomial(probs, 1).item()
            next_token = top_k_indices[next_token_idx].item()
            
            # Stop if EOS token
            if next_token == tokenizer.word2idx.get('<EOS>', -1):
                break
            
            tokens.append(next_token)
    
    # Decode and return
    generated_tokens = tokens[1:]  # Remove BOS
    text = tokenizer.decode(generated_tokens)
    return text


def get_lr_scheduler(optimizer, warmup_steps=1000):
    """Learning rate scheduler with warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return max(0.1, (warmup_steps / step) ** 0.5)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
