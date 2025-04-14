import os
import json
import time
import math
import logging
import argparse
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import save_model

from model import Transformer, ModelArgs, Linear, Block
from inference.model import RMSNorm  # Import explicitly from the model file


logger = logging.getLogger(__name__)


def setup_logging(log_file=None):
    """Configure logging to console and optionally to a file."""
    level = logging.INFO
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
    )

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def setup_distributed(backend="nccl"):
    """Initialize distributed training environment."""
    if torch.distributed.is_initialized():
        return

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend)
        logger.info(f"Initialized distributed training with {world_size} GPUs (rank: {rank}, local_rank: {local_rank})")
    else:
        logger.info("Running in single GPU mode")

    return world_size, rank, local_rank


def get_lr_scheduler(optimizer, warmup_steps, max_steps, min_lr_ratio=0.1):
    """Create a learning rate scheduler with warmup and cosine decay."""
    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def create_fsdp_model(model, rank, world_size, mixed_precision=True):
    """Wrap model with FSDP for distributed training."""
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        CPUOffload,
        BackwardPrefetch,
        ShardingStrategy,
        MixedPrecision,
    )
    
    # Define auto wrap policy
    auto_wrap_policy = transformer_auto_wrap_policy(
        transformer_layer_cls={Block},
    )
    
    # Define mixed precision policy if required
    mp_policy = None
    if mixed_precision:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16
        )
    
    # Wrap the model with FSDP
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=CPUOffload(offload_params=False),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mp_policy,
        device_id=torch.cuda.current_device(),
    )
    
    logger.info(f"Model wrapped with FSDP on rank {rank}")
    return fsdp_model


def compute_loss(logits, labels, ignore_index=-100):
    """Compute cross entropy loss for the model outputs."""
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    # Flatten the tensors
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    
    # Compute loss with possible padding tokens masked
    loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=ignore_index)
    return loss


def save_checkpoint(model, optimizer, scheduler, args, step, epoch, tokenizer=None):
    """Save model checkpoint and optimizer state."""
    os.makedirs(args.save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.save_dir, f"step_{step}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Get model for saving
    if isinstance(model, DDP) or isinstance(model, FSDP):
        model_to_save = model.module
    else:
        model_to_save = model
    
    # Save model weights
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Save using safetensors format
    save_model(model_to_save, os.path.join(checkpoint_dir, f"model{rank}-mp{world_size}.safetensors"))
    
    # Save model configuration
    if rank == 0:
        # Save tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(vars(args.model_args), f, indent=2)
        
        # Save optimizer and scheduler states
        torch.save({
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler else None,
            'epoch': epoch,
            'global_step': step,
        }, os.path.join(checkpoint_dir, "optimizer.pt"))
        
        logger.info(f"Saved checkpoint at step {step} to {checkpoint_dir}")


def get_model_size(model):
    """Calculate model size in millions of parameters."""
    num_params = sum(p.numel() for p in model.parameters())
    return num_params / 1_000_000


def train(args, model, train_dataloader, optimizer, scheduler, tokenizer=None):
    """Main training loop."""
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    # Setup for tracking metrics
    start_time = time.time()
    total_loss = 0.0
    logging_loss = 0.0
    best_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and args.dtype != "fp8")
    
    model.train()
    logger.info(f"Starting training with model of {get_model_size(model):.2f}M parameters")
    
    for epoch in range(args.start_epoch, args.num_epochs):
        if hasattr(train_dataloader, 'sampler') and hasattr(train_dataloader.sampler, 'set_epoch'):
            train_dataloader.sampler.set_epoch(epoch)
        
        for step, batch in enumerate(train_dataloader):
            global_step = epoch * len(train_dataloader) + step
            
            # Move batch to device
            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            # Forward pass with optional mixed precision
            with torch.cuda.amp.autocast(enabled=args.use_amp and args.dtype != "fp8"):
                outputs = model(batch["input_ids"])
                loss = compute_loss(outputs, batch["labels"], ignore_index=tokenizer.pad_token_id if tokenizer else -100)
            
            # Backward pass
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            if args.use_amp and args.dtype != "fp8":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights (maybe)
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    if args.use_amp and args.dtype != "fp8":
                        scaler.unscale_(optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                if args.use_amp and args.dtype != "fp8":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()
            
            # Track loss
            total_loss += loss.item()
            
            # Logging
            if global_step > 0 and global_step % args.logging_steps == 0 and rank == 0:
                # Calculate metrics
                avg_loss = (total_loss - logging_loss) / args.logging_steps
                logging_loss = total_loss
                
                # Calculate time per sample
                elapsed = time.time() - start_time
                samples_per_second = args.logging_steps * args.batch_size * world_size / elapsed
                
                # Get current learning rate
                lr = optimizer.param_groups[0]['lr']
                
                logger.info(
                    f"Epoch: {epoch}, Step: {global_step}, "
                    f"Loss: {avg_loss:.4f}, "
                    f"LR: {lr:.2e}, "
                    f"Samples/sec: {samples_per_second:.2f}"
                )
                start_time = time.time()
            
            # Save checkpoint
            if global_step > 0 and global_step % args.save_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, args,
                    global_step, epoch, tokenizer
                )
                
                # Save best model
                if rank == 0 and avg_loss < best_loss:
                    best_loss = avg_loss
                    best_dir = os.path.join(args.save_dir, "best")
                    if os.path.exists(best_dir):
                        os.system(f"rm -rf {best_dir}")
                    os.system(f"cp -r {os.path.join(args.save_dir, f'step_{global_step}')} {best_dir}")
                    logger.info(f"New best model saved with loss {best_loss:.4f}")
        
        # Save checkpoint at the end of each epoch
        save_checkpoint(
            model, optimizer, scheduler, args,
            global_step, epoch, tokenizer
        )
    
    # Final save at the end of training
    save_checkpoint(
        model, optimizer, scheduler, args,
        global_step, epoch, tokenizer
    )
    
    logger.info("Training complete!")


def main():
    """Main function to setup and start the training process."""
    parser = argparse.ArgumentParser(description="Train DeepSeek-V3 model")
    
    # Model configuration
    parser.add_argument("--config", type=str, required=True, help="Path to model configuration file")
    parser.add_argument("--model_path", type=str, default=None, help="Path to pre-trained model if fine-tuning")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    
    # Training configuration
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data file")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--micro_batch_size", type=int, default=None, help="Micro batch size for gradient accumulation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model hyperparameters
    parser.add_argument("--max_seq_len", type=int, default=None, help="Maximum sequence length")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp8"], help="Model precision")
    parser.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    parser.add_argument("--use_fsdp", action="store_true", help="Use Fully Sharded Data Parallel")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_file)
    
    # Setup distributed training
    world_size, rank, local_rank = setup_distributed()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Set default device
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16 if args.dtype == "bf16" else torch.float8_e4m3fn)
    
    # Load model configuration
    with open(args.config) as f:
        model_args = ModelArgs(**json.load(f))
    
    # Override sequence length if provided
    if args.max_seq_len:
        model_args.max_seq_len = args.max_seq_len
    
    # Set batch size
    if not args.micro_batch_size:
        args.micro_batch_size = args.batch_size
    
    # Store model args
    args.model_args = model_args
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    
    # Create datasets and dataloaders
    from torch.utils.data import IterableDataset, DataLoader, DistributedSampler
    
    # This would typically be replaced by your actual dataset implementation
    # which would be created in start_train.py and imported here
    from start_train import create_train_dataloader
    train_dataloader = create_train_dataloader(
        args.train_file,
        tokenizer,
        args.micro_batch_size,
        max_seq_len=model_args.max_seq_len,
        distributed=(world_size > 1),
        world_size=world_size,
        rank=rank
    )
    
    # Create model
    logger.info(f"Creating model with args: {model_args}")
    model = Transformer(model_args)
    
    # Load pre-trained weights if provided
    if args.model_path:
        from safetensors.torch import load_model
        logger.info(f"Loading pre-trained weights from {args.model_path}")
        load_model(model, os.path.join(args.model_path, f"model{rank}-mp{world_size}.safetensors"))
    
    # Move model to GPU
    model = model.to(f"cuda:{local_rank}")
    
    # Setup distributed training
    if world_size > 1:
        if args.use_fsdp:
            model = create_fsdp_model(model, rank, world_size, mixed_precision=(args.dtype=="bf16"))
        else:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Create optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-5,
    )
    
    # Calculate total training steps
    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    max_steps = args.num_epochs * num_update_steps_per_epoch
    
    # Create LR scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        max_steps=max_steps,
    )
    
    # Log model and training configuration
    if rank == 0:
        logger.info(f"Model config: {model_args}")
        logger.info(f"Training config: {args}")
        logger.info(f"World size: {world_size}, Rank: {rank}")
        logger.info(f"Model parameters: {get_model_size(model):.2f}M")
        logger.info(f"Training for {args.num_epochs} epochs, {max_steps} steps")
    
    # Add properties to args for train function
    args.start_epoch = 0
    
    # Start training
    train(args, model, train_dataloader, optimizer, scheduler, tokenizer)


if __name__ == "__main__":
    main()