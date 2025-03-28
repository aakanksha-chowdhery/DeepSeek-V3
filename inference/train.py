import os
import math
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt

from model import Transformer, ModelArgs

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class DummyDataset(Dataset):
    """Create a dummy dataset for testing"""
    def __init__(self, vocab_size, seq_length, num_samples):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate random input tokens
        x = torch.randint(0, self.vocab_size, (self.seq_length,))
        # For autoregressive LM, targets are the input shifted right
        # (predicting next token at each position)
        y = torch.randint(0, self.vocab_size, (self.seq_length,))
        return x, y


def train_epoch(model, dataloader, optimizer, scheduler, scaler, device, use_amp=True):
    """Train the model for one epoch"""
    model.train()
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    
    for batch_idx, (x, y) in enumerate(dataloader):
        # Move data to device
        x, y = x.to(device), y.to(device)
        batch_size, seq_len = x.size()
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if use_amp:
            with autocast():
                # Modified to handle sequence logits of shape [batch_size, seq_len, vocab_size]
                logits = model(x)  # Expect shape: [batch_size, seq_len, vocab_size]
                
                # Reshape for cross entropy: [batch_size*seq_len, vocab_size]
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = y.view(-1)
                
                # Compute loss across all tokens
                loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard forward pass
            logits = model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Standard backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        # Update learning rate
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item() * batch_size * seq_len
        total_tokens += batch_size * seq_len
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            ms_per_batch = (time.time() - start_time) * 1000 / 10
            cur_loss = total_loss / total_tokens
            ppl = math.exp(cur_loss)
            
            logger.info(
                f"| batch {batch_idx+1:5d}/{len(dataloader):5d} | "
                f"loss {cur_loss:.4f} | ppl {ppl:.2f} | "
                f"ms/batch {ms_per_batch:.2f} | "
                f"lr {scheduler.get_last_lr()[0]:.6f}"
            )
            start_time = time.time()
    
    return total_loss / total_tokens


def validate(model, dataloader, device):
    """Evaluate the model on the validation set"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            batch_size, seq_len = x.size()
            
            # Get sequence logits
            logits = model(x)  # Shape: [batch_size, seq_len, vocab_size]
            
            # Reshape for loss calculation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            
            # Compute loss
            loss = F.cross_entropy(logits_flat, targets_flat)
            
            # Track total loss and token count
            total_loss += loss.item() * batch_size * seq_len
            total_tokens += batch_size * seq_len
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    
    return avg_loss, ppl


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, checkpoint_dir):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    
    logger.info(f"Checkpoint saved to {checkpoint_path}")


def plot_losses(losses, checkpoint_dir):
    """Plot the training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))


def main():
    # Training settings
    batch_size = 4
    seq_length = 128
    num_epochs = 3
    learning_rate = 1e-4
    use_amp = True  # Use mixed precision
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    torch.set_default_dtype(torch.bfloat16)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize model
    logger.info("Initializing model...")
    model_args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=seq_length,
        dtype="bf16",
        vocab_size=102400,  # Use the default from paste.txt
        dim=512,  # Reduced for testing
        inter_dim=2048,  # Reduced for testing
        moe_inter_dim=1024,  # Reduced for testing
        n_layers=4,  # Reduced for testing
        n_dense_layers=1,  # Keep default
        n_heads=8,  # Reduced for testing
        n_routed_experts=8,  # Reduced for testing
        n_shared_experts=2,  # Keep default
        n_activated_experts=4,  # Reduced
        n_expert_groups=1,  # Keep default
        n_limited_groups=1,  # Keep default
        score_func="softmax",  # Keep default
        route_scale=1.0,  # Keep default
        norm_topk_prob=True,  # Keep default
        topk_method="noaux_tc",  # Keep default
        q_lora_rank=0,  # Keep default
        kv_lora_rank=128,  # Reduced for testing
        qk_nope_head_dim=64,  # Reduced for testing
        qk_rope_head_dim=32,  # Reduced for testing
        v_head_dim=64,  # Reduced for testing
        original_seq_len=4096,  # Keep default
        rope_theta=10000.0,  # Keep default
        rope_factor=40,  # Keep default
        beta_fast=32,  # Keep default
        beta_slow=1,  # Keep default
        mscale=1.0,  # Keep default
        mscale_all_dim=1.0,  # Keep default
        aux_loss_alpha=0.0,  # Keep default
        use_seq_aux_loss=False  # Keep default
    )
    
    model = Transformer(model_args).to(device)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")
    
    # Create dummy datasets
    logger.info("Creating synthetic datasets")
    train_dataset = DummyDataset(model_args.vocab_size, seq_length, 100)
    valid_dataset = DummyDataset(model_args.vocab_size, seq_length, 20)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    # Calculate total steps for learning rate scheduler
    total_steps = len(train_loader) * num_epochs
    
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=learning_rate/10)
    
    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join("checkpoints", datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_ppl = validate(model, valid_loader, device)
        val_losses.append(val_loss)
        
        # Log metrics
        logger.info(
            f"| End of epoch {epoch+1:3d} | "
            f"train loss {train_loss:.4f} | "
            f"valid loss {val_loss:.4f} | "
            f"valid ppl {val_ppl:.2f}"
        )
        
        # Save checkpoint
        save_checkpoint(model, optimizer, scheduler, scaler, epoch + 1, val_loss, checkpoint_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            logger.info("New best model saved!")
    
    # Plot losses
    plot_losses(train_losses, checkpoint_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()