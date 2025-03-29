import argparse
import os
import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional

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
import logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainState:
    """State tracking for training progress"""
    step: int = 0
    current_loss: float = -1
    losses: List[float] = field(default_factory=list)
    best_val_loss: float = float('inf')

    def state_dict(self) -> Dict[str, Any]:
        return {
            "step": torch.tensor(self.step, dtype=torch.int32),
            "current_loss": torch.tensor(self.current_loss, dtype=torch.float32),
            "losses": torch.tensor(self.losses, dtype=torch.float32),
            "best_val_loss": torch.tensor(self.best_val_loss, dtype=torch.float32),
        }

    def load_state_dict(self, state_dict) -> None:
        self.step = state_dict["step"].item()
        self.current_loss = state_dict["current_loss"].item()
        self.losses = state_dict["losses"].tolist()
        self.best_val_loss = state_dict["best_val_loss"].item()


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


def build_optimizer(model, args):
    """Build optimizer based on arguments"""
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} not implemented")
    
    return optimizer


def build_scheduler(optimizer, args, total_steps):
    """Build learning rate scheduler"""
    return CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr/10)


def build_grad_scaler(args):
    """Build gradient scaler for mixed precision training"""
    return GradScaler() if args.use_amp else None


def train_step(
    model, 
    batch, 
    optimizer, 
    scaler, 
    device, 
    use_amp=False
):
    """Perform a single training step"""
    x, y = batch
    x, y = x.to(device), y.to(device)
    batch_size, seq_len = x.size()
    
    # Zero the gradients
    optimizer.zero_grad(set_to_none=True)
    
    # Forward pass with mixed precision
    if use_amp:
        with autocast():
            logits = model(x)
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y.view(-1)
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        optimizer.step()
    
    # Return loss and number of tokens
    return loss.item(), batch_size * seq_len


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
            logits = model(x)
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


def save_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    scaler, 
    train_state, 
    checkpoint_dir, 
    is_best=False
):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{train_state.step}.pt")
    
    # Build checkpoint data
    checkpoint = {
        'step': train_state.step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_state': train_state.state_dict(),
    }
    
    if scaler is not None:
        checkpoint['scaler_state_dict'] = scaler.state_dict()
    
    # Save the checkpoint
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save best model separately if it's the best so far
    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pt")
        torch.save(model.state_dict(), best_path)
        logger.info("New best model saved!")


def load_checkpoint(
    model, 
    optimizer, 
    scheduler, 
    scaler, 
    train_state, 
    checkpoint_path
):
    """Load model checkpoint"""
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path}, starting from scratch")
        return
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    if 'train_state' in checkpoint:
        train_state.load_state_dict(checkpoint['train_state'])
    else:
        train_state.step = checkpoint['step']
    
    if scaler is not None and 'scaler_state_dict' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
    
    logger.info(f"Loaded checkpoint from step {train_state.step}")


def plot_losses(losses, checkpoint_dir):
    """Plot the training losses"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.savefig(os.path.join(checkpoint_dir, 'training_loss.png'))


def get_model_args(args):
    """Build model arguments"""
    return ModelArgs(
        max_batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        vocab_size=args.vocab_size,
        dim=args.dim,
        inter_dim=args.inter_dim,
        moe_inter_dim=args.moe_inter_dim,
        n_layers=args.n_layers,
        n_dense_layers=args.n_dense_layers,
        n_heads=args.n_heads,
        n_routed_experts=args.n_routed_experts,
        n_shared_experts=args.n_shared_experts,
        n_activated_experts=args.n_activated_experts,
        n_expert_groups=args.n_expert_groups,
        n_limited_groups=args.n_limited_groups,
        score_func=args.score_func,
        route_scale=args.route_scale,
        norm_topk_prob=args.norm_topk_prob,
        topk_method=args.topk_method,
        q_lora_rank=args.q_lora_rank,
        kv_lora_rank=args.kv_lora_rank,
        qk_nope_head_dim=args.qk_nope_head_dim,
        qk_rope_head_dim=args.qk_rope_head_dim,
        v_head_dim=args.v_head_dim,
        original_seq_len=args.original_seq_len,
        rope_theta=args.rope_theta,
        rope_factor=args.rope_factor,
        beta_fast=args.beta_fast,
        beta_slow=args.beta_slow,
        mscale=args.mscale,
        mscale_all_dim=args.mscale_all_dim,
        aux_loss_alpha=args.aux_loss_alpha,
        use_seq_aux_loss=args.use_seq_aux_loss,
    )


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set precision
    if args.dtype == "bf16":
        torch.set_default_dtype(torch.bfloat16)
    elif args.dtype == "fp16":
        torch.set_default_dtype(torch.float16)
    else:
        torch.set_default_dtype(torch.float32)
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Initialize model with arguments
    logger.info("Initializing model...")
    model_args = get_model_args(args)
    model = Transformer(model_args).to(device)
    
    # Calculate model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model initialized with {total_params:,} parameters")
    
    # Create datasets
    logger.info("Creating datasets")
    train_dataset = DummyDataset(args.vocab_size, args.max_seq_len, args.train_samples)
    valid_dataset = DummyDataset(args.vocab_size, args.max_seq_len, args.valid_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Calculate total steps
    total_steps = len(train_loader) * args.epochs
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, args)
    scheduler = build_scheduler(optimizer, args, total_steps)
    
    # Initialize gradient scaler for mixed precision
    scaler = build_grad_scaler(args)
    
    # Initialize training state
    train_state = TrainState()
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.checkpoint_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load checkpoint if provided
    if args.resume_from:
        load_checkpoint(model, optimizer, scheduler, scaler, train_state, args.resume_from)
    
    # Training loop
    logger.info("Starting training...")
    
    # Initialize metrics tracking
    train_losses = []
    val_losses = []
    losses_since_last_log = []
    total_tokens_since_last_log = 0
    time_last_log = time.time()
    
    model.train()
    while train_state.step < total_steps:
        # Get batch
        for batch_idx, batch in enumerate(train_loader):
            train_state.step += 1
            
            # Train step
            loss, n_tokens = train_step(model, batch, optimizer, scaler, device, args.use_amp)
            
            # Update learning rate
            scheduler.step()
            
            # Track metrics
            train_state.current_loss = loss
            train_state.losses.append(loss)
            losses_since_last_log.append(loss)
            total_tokens_since_last_log += n_tokens
            
            # Log progress
            if train_state.step % args.log_interval == 0:
                time_now = time.time()
                time_delta = time_now - time_last_log
                tokens_per_sec = total_tokens_since_last_log / time_delta
                avg_loss = sum(losses_since_last_log) / len(losses_since_last_log)
                ppl = math.exp(avg_loss)
                
                logger.info(
                    f"Step {train_state.step}/{total_steps} | "
                    f"Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | "
                    f"Tokens/sec: {tokens_per_sec:.1f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f}"
                )
                
                # Reset metrics
                losses_since_last_log = []
                total_tokens_since_last_log = 0
                time_last_log = time_now
            
            # Validate and save checkpoint
            if train_state.step % args.eval_interval == 0:
                val_loss, val_ppl = validate(model, valid_loader, device)
                val_losses.append(val_loss)
                
                logger.info(
                    f"Validation | Step {train_state.step}/{total_steps} | "
                    f"Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
                )
                
                # Save checkpoint
                is_best = val_loss < train_state.best_val_loss
                if is_best:
                    train_state.best_val_loss = val_loss
                
                save_checkpoint(
                    model, optimizer, scheduler, scaler, train_state, 
                    checkpoint_dir, is_best=is_best
                )
            
            # Check if we've reached the total number of steps
            if train_state.step >= total_steps:
                break
    
    # Final validation
    val_loss, val_ppl = validate(model, valid_loader, device)
    val_losses.append(val_loss)
    
    logger.info(
        f"Final Validation | Loss: {val_loss:.4f} | PPL: {val_ppl:.2f}"
    )
    
    # Save final model
    is_best = val_loss < train_state.best_val_loss
    if is_best:
        train_state.best_val_loss = val_loss
    
    save_checkpoint(
        model, optimizer, scheduler, scaler, train_state, 
        checkpoint_dir, is_best=is_best
    )
    
    # Plot losses
    plot_losses(train_state.losses, checkpoint_dir)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLM model")
    
    # Model configuration
    parser.add_argument("--vocab_size", type=int, default=102400, help="Vocabulary size")
    parser.add_argument("--dim", type=int, default=512, help="Model dimension")
    parser.add_argument("--inter_dim", type=int, default=2048, help="Intermediate dimension")
    parser.add_argument("--moe_inter_dim", type=int, default=1024, help="MoE intermediate dimension")
    parser.add_argument("--n_layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--n_dense_layers", type=int, default=1, help="Number of dense layers")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_routed_experts", type=int, default=8, help="Number of routed experts")
    parser.add_argument("--n_shared_experts", type=int, default=2, help="Number of shared experts")
    parser.add_argument("--n_activated_experts", type=int, default=4, help="Number of activated experts")
    parser.add_argument("--n_expert_groups", type=int, default=1, help="Number of expert groups")
    parser.add_argument("--n_limited_groups", type=int, default=1, help="Number of limited groups")
    parser.add_argument("--score_func", type=str, default="softmax", help="Score function for MoE")
    parser.add_argument("--route_scale", type=float, default=1.0, help="Route scale for MoE")
    parser.add_argument("--norm_topk_prob", type=bool, default=True, help="Normalize top-k probabilities")
    parser.add_argument("--topk_method", type=str, default="noaux_tc", help="Top-k method for MoE")
    parser.add_argument("--q_lora_rank", type=int, default=0, help="Query LoRA rank")
    parser.add_argument("--kv_lora_rank", type=int, default=128, help="Key-value LoRA rank")
    parser.add_argument("--qk_nope_head_dim", type=int, default=64, help="QK no-PE head dimension")
    parser.add_argument("--qk_rope_head_dim", type=int, default=32, help="QK RoPE head dimension")
    parser.add_argument("--v_head_dim", type=int, default=64, help="Value head dimension")
    parser.add_argument("--original_seq_len", type=int, default=4096, help="Original sequence length")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta")
    parser.add_argument("--rope_factor", type=float, default=40, help="RoPE factor")
    parser.add_argument("--beta_fast", type=int, default=32, help="Fast beta correction factor")
    parser.add_argument("--beta_slow", type=int, default=1, help="Slow beta correction factor")
    parser.add_argument("--mscale", type=float, default=1.0, help="Scaling factor for extended attention")
    parser.add_argument("--mscale_all_dim", type=float, default=1.0, help="Scaling factor all dimensions")
    parser.add_argument("--aux_loss_alpha", type=float, default=0.0, help="Auxiliary loss alpha")
    parser.add_argument("--use_seq_aux_loss", type=bool, default=False, help="Use sequence auxiliary loss")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Max norm for gradient clipping")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train_samples", type=int, default=100, help="Number of training samples")
    parser.add_argument("--valid_samples", type=int, default=20, help="Number of validation samples")
    parser.add_argument("--optimizer", type=str, default="AdamW", help="Optimizer (Adam or AdamW)")
    
    # System configuration
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], 
                        help="Data type for training")
    
    # Logging and checkpointing
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval in steps")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval in steps")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", 
                        help="Directory for saving checkpoints")
    parser.add_argument("--resume_from", type=str, default="", 
                        help="Resume training from checkpoint path")
    
    args = parser.parse_args()
    main(args)