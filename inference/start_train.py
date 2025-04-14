import os
import json
import logging
from typing import Dict, List, Optional, Union, Iterator, Tuple
import multiprocessing
from functools import partial

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class TokenizerWrapper:
    """Wrapper around tokenizer to provide consistent interface."""
    
    def __init__(
        self, 
        tokenizer_path: str,
        max_seq_len: int = 4096,
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.max_seq_len = max_seq_len
        self.add_bos = add_bos
        self.add_eos = add_eos
        
        # Set padding token if not set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
        logger.info(f"Loaded tokenizer from {tokenizer_path}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
    
    def __call__(self, text: str, padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize text with options for padding and truncation."""
        encodings = self.tokenizer(
            text,
            add_special_tokens=False,  # We'll add special tokens manually
            padding=padding,
            truncation=truncation,
            max_length=self.max_seq_len - (self.add_bos + self.add_eos),  # Allow space for special tokens
            return_tensors="pt",
        )
        
        input_ids = encodings["input_ids"]
        
        # Add BOS token if required
        if self.add_bos:
            bos_tensor = torch.full((input_ids.size(0), 1), self.tokenizer.bos_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([bos_tensor, input_ids], dim=1)
            
        # Add EOS token if required
        if self.add_eos:
            eos_tensor = torch.full((input_ids.size(0), 1), self.tokenizer.eos_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, eos_tensor], dim=1)
        
        encodings["input_ids"] = input_ids
        # Update attention mask if it exists
        if "attention_mask" in encodings:
            attention_mask = encodings["attention_mask"]
            if self.add_bos:
                bos_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype)
                attention_mask = torch.cat([bos_mask, attention_mask], dim=1)
            if self.add_eos:
                eos_mask = torch.ones((attention_mask.size(0), 1), dtype=attention_mask.dtype)
                attention_mask = torch.cat([attention_mask, eos_mask], dim=1)
            encodings["attention_mask"] = attention_mask
            
        return encodings
    
    def tokenize_for_training(self, text: str) -> Dict[str, torch.Tensor]:
        """Create input_ids and labels for language modeling."""
        encodings = self(text)
        result = {
            "input_ids": encodings["input_ids"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0).clone(),
        }
        if "attention_mask" in encodings:
            result["attention_mask"] = encodings["attention_mask"].squeeze(0)
        return result
    
    def chat_tokenize(self, messages: List[Dict[str, str]], add_generation_prompt: bool = True) -> Dict[str, torch.Tensor]:
        """Tokenize chat messages using the chat template."""
        prompt_tokens = self.tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=add_generation_prompt,
            return_tensors="pt"
        )
        
        result = {
            "input_ids": prompt_tokens,
            "labels": prompt_tokens.clone(),
        }
        
        # Create attention mask
        attention_mask = torch.ones_like(prompt_tokens)
        result["attention_mask"] = attention_mask
        
        return result


class TextDataset(Dataset):
    """Dataset for pre-tokenized text data."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerWrapper,
        max_seq_len: int = 4096,
        cache_tokenization: bool = False,
    ):
        """
        Args:
            file_path: Path to text file with one example per line
            tokenizer: TokenizerWrapper instance
            max_seq_len: Maximum sequence length
            cache_tokenization: Whether to cache tokenized examples
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_tokenization = cache_tokenization
        
        # Load text data
        logger.info(f"Loading text data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.examples = [line.strip() for line in f if line.strip()]
        
        logger.info(f"Loaded {len(self.examples)} examples")
        
        # Tokenize and cache if enabled
        self.tokenized_examples = None
        if self.cache_tokenization:
            logger.info("Pre-tokenizing all examples (this might take a while)...")
            self.tokenized_examples = []
            for ex in self.examples:
                self.tokenized_examples.append(self.tokenizer.tokenize_for_training(ex))
            logger.info("Tokenization complete.")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized example."""
        if self.tokenized_examples is not None:
            return self.tokenized_examples[idx]
        
        text = self.examples[idx]
        return self.tokenizer.tokenize_for_training(text)


class StreamingTextDataset(IterableDataset):
    """Dataset for streaming text data without loading everything into memory."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerWrapper,
        max_seq_len: int = 4096,
        buffer_size: int = 1000,
        shuffle: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Args:
            file_path: Path to text file with one example per line
            tokenizer: TokenizerWrapper instance
            max_seq_len: Maximum sequence length
            buffer_size: Size of the buffer for shuffling
            shuffle: Whether to shuffle examples
            world_size: Number of processes for distributed training
            rank: Process rank for distributed training
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        
        # Get an estimate of dataset size for planning
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Count lines in first 1MB
                sample = f.read(1024 * 1024)
                if not sample:
                    estimated_lines = 0
                else:
                    estimated_lines = sample.count('\n')
                    file_size = os.path.getsize(file_path)
                    # Simple estimate based on first chunk
                    if file_size > 0:
                        estimated_lines = int(estimated_lines * (file_size / len(sample.encode('utf-8'))))
            
            self.estimated_size = estimated_lines
            logger.info(f"Estimated dataset size: {self.estimated_size} examples")
        except Exception as e:
            logger.warning(f"Could not estimate dataset size: {e}")
            self.estimated_size = None
    
    def estimate_len(self) -> int:
        """Get estimated dataset length."""
        if self.estimated_size is None:
            return 10000  # Default estimate
        return self.estimated_size
    
    def process_chunk(self, chunk: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
        """Process a chunk of text examples."""
        for text in chunk:
            if not text.strip():
                continue
            yield self.tokenizer.tokenize_for_training(text.strip())
    
    def read_with_sharding(self) -> Iterator[str]:
        """Read file with distributed sharding."""
        worker_info = torch.utils.data.get_worker_info()
        num_workers_per_rank = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        
        # Calculate global worker ID
        global_worker_id = self.rank * num_workers_per_rank + worker_id
        global_num_workers = self.world_size * num_workers_per_rank
        
        logger.debug(f"Worker {global_worker_id}/{global_num_workers} starting")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % global_num_workers == global_worker_id:
                    yield line
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator for streaming text data."""
        if self.shuffle:
            # Buffer for shuffling
            buffer = []
            
            for line in self.read_with_sharding():
                buffer.append(line)
                if len(buffer) >= self.buffer_size:
                    # Shuffle buffer
                    import random
                    random.shuffle(buffer)
                    # Process and yield examples
                    yield from self.process_chunk(buffer)
                    buffer = []
            
            # Process remaining examples
            if buffer:
                import random
                random.shuffle(buffer)
                yield from self.process_chunk(buffer)
        else:
            # Process examples in order
            yield from self.process_chunk(self.read_with_sharding())


class ChatDataset(Dataset):
    """Dataset for chat conversations."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerWrapper,
        max_seq_len: int = 4096,
    ):
        """
        Args:
            file_path: Path to JSON file with chat conversations
            tokenizer: TokenizerWrapper instance
            max_seq_len: Maximum sequence length
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Load chat data
        logger.info(f"Loading chat data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.conversations = json.load(f)
        
        logger.info(f"Loaded {len(self.conversations)} conversations")
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a tokenized conversation."""
        conversation = self.conversations[idx]
        return self.tokenizer.chat_tokenize(conversation)


class StreamingChatDataset(IterableDataset):
    """Dataset for streaming chat conversations."""
    
    def __init__(
        self,
        file_path: str,
        tokenizer: TokenizerWrapper,
        max_seq_len: int = 4096,
        buffer_size: int = 1000,
        shuffle: bool = True,
        world_size: int = 1,
        rank: int = 0,
    ):
        """
        Args:
            file_path: Path to JSONL file with chat conversations (one per line)
            tokenizer: TokenizerWrapper instance
            max_seq_len: Maximum sequence length
            buffer_size: Size of the buffer for shuffling
            shuffle: Whether to shuffle examples
            world_size: Number of processes for distributed training
            rank: Process rank for distributed training
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.world_size = world_size
        self.rank = rank
        
        # Get an estimate of dataset size for planning
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Count lines in first 1MB
                sample = f.read(1024 * 1024)
                if not sample:
                    estimated_lines = 0
                else:
                    estimated_lines = sample.count('\n')
                    file_size = os.path.getsize(file_path)
                    # Simple estimate based on first chunk
                    if file_size > 0:
                        estimated_lines = int(estimated_lines * (file_size / len(sample.encode('utf-8'))))
            
            self.estimated_size = estimated_lines
            logger.info(f"Estimated dataset size: {self.estimated_size} conversations")
        except Exception as e:
            logger.warning(f"Could not estimate dataset size: {e}")
            self.estimated_size = None
    
    def estimate_len(self) -> int:
        """Get estimated dataset length."""
        if self.estimated_size is None:
            return 10000  # Default estimate
        return self.estimated_size
    
    def process_chunk(self, chunk: List[str]) -> Iterator[Dict[str, torch.Tensor]]:
        """Process a chunk of conversations."""
        for line in chunk:
            if not line.strip():
                continue
            try:
                conversation = json.loads(line.strip())
                yield self.tokenizer.chat_tokenize(conversation)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from line: {line[:50]}...")
    
    def read_with_sharding(self) -> Iterator[str]:
        """Read file with distributed sharding."""
        worker_info = torch.utils.data.get_worker_info()
        num_workers_per_rank = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        
        # Calculate global worker ID
        global_worker_id = self.rank * num_workers_per_rank + worker_id
        global_num_workers = self.world_size * num_workers_per_rank
        
        logger.debug(f"Worker {global_worker_id}/{global_num_workers} starting")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i % global_num_workers == global_worker_id:
                    yield line
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator for streaming chat data."""
        if self.shuffle:
            # Buffer for shuffling
            buffer = []
            
            for line in self.read_with_sharding():
                buffer.append(line)
                if len(buffer) >= self.buffer_size:
                    # Shuffle buffer
                    import random
                    random.shuffle(buffer)
                    # Process and yield examples
                    yield from self.process_chunk(buffer)
                    buffer = []
            
            # Process remaining examples
            if buffer:
                import random
                random.shuffle(buffer)
                yield from self.process_chunk(buffer)
        else:
            # Process examples in order
            yield from self.process_chunk(self.read_with_sharding())


def get_tokenizer(tokenizer_path: str, max_seq_len: int = 4096) -> TokenizerWrapper:
    """Get a TokenizerWrapper instance."""
    return TokenizerWrapper(
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
    )


def create_dataset(
    file_path: str,
    tokenizer: TokenizerWrapper,
    max_seq_len: int = 4096,
    dataset_type: str = "text",
    streaming: bool = False,
    cache_tokenization: bool = False,
    buffer_size: int = 1000,
    shuffle: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> Union[Dataset, IterableDataset]:
    """Create a dataset based on file type and options."""
    # Check file extension
    if file_path.endswith(".txt") or dataset_type == "text":
        if streaming:
            return StreamingTextDataset(
                file_path=file_path,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                buffer_size=buffer_size,
                shuffle=shuffle,
                world_size=world_size,
                rank=rank,
            )
        else:
            return TextDataset(
                file_path=file_path,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                cache_tokenization=cache_tokenization,
            )
    elif file_path.endswith((".json", ".jsonl")) or dataset_type == "chat":
        if streaming:
            return StreamingChatDataset(
                file_path=file_path,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                buffer_size=buffer_size,
                shuffle=shuffle,
                world_size=world_size,
                rank=rank,
            )
        else:
            return ChatDataset(
                file_path=file_path,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
            )
    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def create_dataloader(
    dataset: Union[Dataset, IterableDataset],
    batch_size: int,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    if isinstance(dataset, IterableDataset):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
    else:
        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            sampler = None
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )


def create_train_dataloader(
    file_path: str,
    tokenizer_or_path: Union[str, TokenizerWrapper],
    batch_size: int,
    max_seq_len: int = 4096,
    dataset_type: str = "auto",
    streaming: bool = True,
    cache_tokenization: bool = False,
    buffer_size: int = 1000,
    shuffle: bool = True,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for training."""
    # Determine dataset type if auto
    if dataset_type == "auto":
        if file_path.endswith(".txt"):
            dataset_type = "text"
        elif file_path.endswith((".json", ".jsonl")):
            dataset_type = "chat"
        else:
            logger.warning(f"Could not determine dataset type from file extension, defaulting to 'text'")
            dataset_type = "text"
    
    # Get tokenizer if path is provided
    if isinstance(tokenizer_or_path, str):
        tokenizer = get_tokenizer(tokenizer_or_path, max_seq_len)
    else:
        tokenizer = tokenizer_or_path
    
    # Create dataset
    dataset = create_dataset(
        file_path=file_path,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        dataset_type=dataset_type,
        streaming=streaming,
        cache_tokenization=cache_tokenization,
        buffer_size=buffer_size,
        shuffle=shuffle,
        world_size=world_size,
        rank=rank,
    )
    
    # Create dataloader
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_validation_dataloader(
    file_path: str,
    tokenizer_or_path: Union[str, TokenizerWrapper],
    batch_size: int,
    max_seq_len: int = 4096,
    dataset_type: str = "auto",
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create a DataLoader for validation."""
    # We don't stream validation data since we want to compute metrics on the full set
    return create_train_dataloader(
        file_path=file_path,
        tokenizer_or_path=tokenizer_or_path,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        dataset_type=dataset_type,
        streaming=False,  # Don't stream validation data
        cache_tokenization=True,  # Cache tokenization for validation
        shuffle=False,  # Don't shuffle validation data
        distributed=distributed,
        world_size=world_size,
        rank=rank,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def create_deepseek_v3_tokenizer(tokenizer_path: str, max_seq_len: int = 4096) -> TokenizerWrapper:
    """Create a TokenizerWrapper instance specifically configured for DeepSeek-V3."""
    return TokenizerWrapper(
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        add_bos=True,
        add_eos=True,
    )


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Test tokenizer and dataset creation")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer")
    parser.add_argument("--data_file", type=str, required=True, help="Path to data file")
    parser.add_argument("--max_seq_len", type=int, default=4096, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    
    # Create tokenizer
    tokenizer = get_tokenizer(args.tokenizer_path, args.max_seq_len)
    
    # Create dataloader
    dataloader = create_train_dataloader(
        file_path=args.data_file,
        tokenizer_or_path=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
    )
    
    # Test dataloader
    logger.info("Testing dataloader...")
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Only look at first few batches
            break
        logger.info(f"Batch {i}: {batch['input_ids'].shape}")
        
    logger.info("Test complete!")