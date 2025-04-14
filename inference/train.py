#!/usr/bin/env python
import os
import json
import time
import math
import logging
import argparse

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from safetensors.torch import save_model

from model import Transformer, ModelArgs
from start_train import TokenizerWrapper, StreamingTextDataset, TextDataset

logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )

def setup_dist():
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank       = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        logger.info(f"DDP: rank {rank}/{world_size} on GPU {local_rank}")
    return world_size, rank, local_rank

def get_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def make_dataloader(txt_file, tokenizer, seq_len, batch_size, streaming, world_size, rank, num_workers):
    if streaming:
        ds = StreamingTextDataset(txt_file, tokenizer, max_seq_len=seq_len,
                                  buffer_size=1000, shuffle=True,
                                  world_size=world_size, rank=rank)
    else:
        ds = TextDataset(txt_file, tokenizer, max_seq_len=seq_len, cache_tokenization=False)
    if isinstance(ds, IterableDataset):
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    else:
        sampler = torch.utils.data.distributed.DistributedSampler(ds) if world_size>1 else None
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, pin_memory=True)

def save_checkpoint(model, optimizer, scheduler, args, step, rank):
    ckpt_dir = os.path.join(args.save_dir, f"step_{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    # unwrap
    mod = model.module if isinstance(model, (DDP, FSDP)) else model
    save_model(mod, os.path.join(ckpt_dir, f"model{rank}-mp{args.world_size}.safetensors"))
    if rank==0:
        torch.save({
            "opt": optimizer.state_dict(),
            "sch": scheduler.state_dict(),
            "step": step,
        }, os.path.join(ckpt_dir, "optim.pt"))
        with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
            json.dump(vars(args.model_args), f, indent=2)
        logger.info(f"Checkpoint saved at step {step}")

def train_loop(args, model, dataloader, optimizer, scheduler, tokenizer, rank):
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and args.dtype=="bf16")
    total_steps = args.num_epochs * len(dataloader) // args.grad_accum
    logger.info(f"Total steps: {total_steps}, epochs: {args.num_epochs}")

    global_step = 0
    model.train()
    for epoch in range(args.num_epochs):
        if hasattr(dataloader, "sampler") and isinstance(dataloader.sampler, torch.utils.data.Sampler):
            dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            global_step += 1
            # move to GPU
            batch = {k: v.cuda() for k,v in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.use_amp and args.dtype=="bf16"):
                logits = model(batch["input_ids"])
                # next-token loss
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = batch["labels"][..., 1:].contiguous()
                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=tokenizer.tokenizer.pad_token_id
                ) / args.grad_accum

            if args.use_amp and args.dtype=="bf16":
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if global_step % args.grad_accum == 0:
                if args.max_grad_norm>0:
                    if args.use_amp and args.dtype=="bf16":
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                if args.use_amp and args.dtype=="bf16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if rank==0 and global_step % args.log_steps==0:
                lr = optimizer.param_groups[0]["lr"]
                logger.info(f"Epoch {epoch} Step {global_step} Loss {loss.item()*args.grad_accum:.4f} LR {lr:.2e}")

            if rank==0 and global_step % args.save_steps==0:
                save_checkpoint(model, optimizer, scheduler, args, global_step, rank)

    # final save
    if rank==0:
        save_checkpoint(model, optimizer, scheduler, args, global_step, rank)
    logger.info("Training complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",         type=str, required=True)
    parser.add_argument("--tokenizer_path",type=str, required=True)
    parser.add_argument("--train_file",     type=str, required=True)
    parser.add_argument("--save_dir",       type=str, required=True)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--num_epochs",     type=int, default=3)
    parser.add_argument("--learning_rate",  type=float, default=5e-5)
    parser.add_argument("--warmup_steps",   type=int, default=100)
    parser.add_argument("--log_steps",      type=int, default=50)
    parser.add_argument("--save_steps",     type=int, default=500)
    parser.add_argument("--grad_accum",     type=int, default=1)
    parser.add_argument("--max_grad_norm",  type=float, default=1.0)
    parser.add_argument("--use_amp",        action="store_true")
    parser.add_argument("--dtype",          choices=["bf16","fp8"], default="bf16")
    parser.add_argument("--streaming",      action="store_true")
    parser.add_argument("--num_workers",    type=int, default=4)
    args = parser.parse_args()

    setup_logging()
    world_size, rank, local_rank = setup_dist()
    args.world_size = world_size
    torch.manual_seed(42)
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16 if args.dtype=="bf16" else torch.float32)

    # load config
    with open(args.config) as f:
        args.model_args = ModelArgs(**json.load(f))

    # build model
    model = Transformer(args.model_args).cuda()
    if world_size>1:
        # FSDP wrap
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload, BackwardPrefetch, ShardingStrategy, MixedPrecision
        auto_wrap = transformer_auto_wrap_policy(transformer_layer_cls={})
        mp_policy = MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16) if args.dtype=="bf16" else None
        model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD,
                     auto_wrap_policy=auto_wrap,
                     backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
                     cpu_offload=CPUOffload(offload_params=False),
                     mixed_precision=mp_policy,
                     device_id=local_rank)
    else:
        model = DDP(model, device_ids=[local_rank])

    # tokenizer + data
    tokenizer = TokenizerWrapper(args.tokenizer_path, max_seq_len=args.model_args.max_seq_len)
    dataloader = make_dataloader(
        args.train_file, tokenizer, args.model_args.max_seq_len,
        args.batch_size, args.streaming, world_size, rank, args.num_workers
    )

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    total_steps = args.num_epochs * len(dataloader) // args.grad_accum
    scheduler = get_scheduler(optimizer, args.warmup_steps, total_steps)

    # train!
    train_loop(args, model, dataloader, optimizer, scheduler, tokenizer, rank)

if __name__=="__main__":
    main()
