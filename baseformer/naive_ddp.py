"""
Naive Distributed Data Parallel (DDP) implementation.

Demonstrates basic gradient synchronization across multiple GPUs without
overlapping communication with computation.

Usage:
    uv run python -m baseformer.naive_ddp                   # DDP mode
    uv run python -m baseformer.naive_ddp --no_distributed  # Single GPU mode

Profiling:
    nsys profile \
        --trace=cuda,nvtx,osrt \
        --cuda-memory-usage=true \
        --gpu-metrics-device=all \
        --output=naive_ddp_profile_timing \
        uv run python -m baseformer.naive_ddp
"""

import argparse
import os
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import torch
import torch.cuda.nvtx as nvtx
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from baseformer.nn.position import RotaryPositionalEmbedding
from baseformer.optim.adamw import AdamW
from baseformer.scripts.utils import load_model, VOCAB_SIZE


# Training configuration
SEED = 42
BATCH_SIZE = 16
SEQ_LEN = 256
NUM_STEPS = 16
LR = 1e-4
CHECKPOINT_DIR = Path("checkpoints/naive_ddp")


def setup(rank: int, world_size: int):
    """Initialize distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12950"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Destroy distributed process group."""
    dist.barrier()
    dist.destroy_process_group()


def create_dataset(seed: int, num_samples: int, seq_len: int, vocab_size: int) -> torch.Tensor:
    """Create deterministic random dataset for reproducibility."""
    torch.manual_seed(seed)
    return torch.randint(0, vocab_size, (num_samples, seq_len))


def sync_gradients(model: nn.Module):
    """All-reduce gradients across all ranks with mean operation."""
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.grad is not None:
            with nvtx.range("all_reduce"):
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad.div_(world_size)


def train_single_gpu(model: nn.Module):
    """Single GPU training (no distributed)."""
    device = torch.device("cuda:0")
    model = model.to(device)
    
    # Recreate RoPE on the correct device
    if model.rope is not None:
        d_k = model.d_model // model.num_heads
        model.rope = RotaryPositionalEmbedding(
            theta=10000.0,
            d_k=d_k,
            max_seq_len=SEQ_LEN,
            device=device,
        )
    
    # Create deterministic dataset
    dataset = create_dataset(SEED, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    data = dataset.to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=LR)
    
    print("Training on single GPU (no distributed)")
    print(f"Samples: {BATCH_SIZE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Timing accumulators
    step_times: dict[str, list[float]] = defaultdict(list)
    
    # Training loop
    for step in range(NUM_STEPS):
        with nvtx.range(f"step_{step}"):
            # Synchronize before starting step timing
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            
            with nvtx.range("zero_grad"):
                optimizer.zero_grad()
            torch.cuda.synchronize()
            zero_grad_time = time.perf_counter()
            
            # Forward pass - use input tokens as targets shifted by 1
            inputs = data[:, :-1]
            targets = data[:, 1:]
            
            with nvtx.range("forward"):
                logits = model(inputs)
            torch.cuda.synchronize()
            forward_time = time.perf_counter()
            
            with nvtx.range("loss"):
                loss = nn.functional.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    targets.reshape(-1),
                )
            torch.cuda.synchronize()
            loss_time = time.perf_counter()
            
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()
            backward_time = time.perf_counter()
            
            with nvtx.range("optimizer_step"):
                optimizer.step()
            torch.cuda.synchronize()
            optimizer_time = time.perf_counter()
            
            # Record times
            step_times["zero_grad"].append(zero_grad_time - step_start)
            step_times["forward"].append(forward_time - zero_grad_time)
            step_times["loss"].append(loss_time - forward_time)
            step_times["backward"].append(backward_time - loss_time)
            step_times["optimizer_step"].append(optimizer_time - backward_time)
            step_times["total"].append(optimizer_time - step_start)
            
            print(f"Step {step + 1}/{NUM_STEPS}, Loss: {loss.item():.4f}, "
                  f"Time: {(optimizer_time - step_start) * 1000:.2f}ms")
    
    # Print average timing statistics
    print("\n--- Average Step Timings (ms) ---")
    for name, times in step_times.items():
        avg_ms = sum(times) / len(times) * 1000
        print(f"  {name}: {avg_ms:.2f}ms")
    print("---------------------------------\n")
    
    # Save checkpoint
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "final_single_gpu.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": NUM_STEPS,
        "loss": loss.item(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


def train_ddp(rank: int, world_size: int, model: nn.Module):
    """Per-rank training function for DDP."""
    setup(rank, world_size)
    
    # Move model copy to this rank's GPU
    device = torch.device(f"cuda:{rank}")
    model = deepcopy(model).to(device)
    
    # Recreate RoPE on the correct device
    if model.rope is not None:
        d_k = model.d_model // model.num_heads
        model.rope = RotaryPositionalEmbedding(
            theta=10000.0,
            d_k=d_k,
            max_seq_len=SEQ_LEN,
            device=device,
        )
    
    # Create deterministic dataset (same on all ranks)
    dataset = create_dataset(SEED, BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    
    # Each rank gets a disjoint shard
    samples_per_rank = BATCH_SIZE // world_size
    start_idx = rank * samples_per_rank
    end_idx = start_idx + samples_per_rank
    local_data = dataset[start_idx:end_idx].to(device)
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=LR)
    
    if rank == 0:
        print(f"Training with {world_size} GPUs (DDP)")
        print(f"Samples per rank: {samples_per_rank}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Timing accumulators
    step_times: dict[str, list[float]] = defaultdict(list)
    
    # Training loop
    for step in range(NUM_STEPS):
        with nvtx.range(f"step_{step}"):
            # Synchronize before starting step timing
            torch.cuda.synchronize()
            step_start = time.perf_counter()
            
            with nvtx.range("zero_grad"):
                optimizer.zero_grad()
            torch.cuda.synchronize()
            zero_grad_time = time.perf_counter()
            
            # Forward pass - use input tokens as targets shifted by 1
            inputs = local_data[:, :-1]
            targets = local_data[:, 1:]
            
            with nvtx.range("forward"):
                logits = model(inputs)
            torch.cuda.synchronize()
            forward_time = time.perf_counter()
            
            with nvtx.range("loss"):
                loss = nn.functional.cross_entropy(
                    logits.view(-1, model.vocab_size),
                    targets.reshape(-1),
                )
            torch.cuda.synchronize()
            loss_time = time.perf_counter()
            
            with nvtx.range("backward"):
                loss.backward()
            torch.cuda.synchronize()
            backward_time = time.perf_counter()
            
            with nvtx.range("sync_gradients"):
                sync_gradients(model)
            torch.cuda.synchronize()
            sync_time = time.perf_counter()
            
            with nvtx.range("optimizer_step"):
                optimizer.step()
            torch.cuda.synchronize()
            optimizer_time = time.perf_counter()
            
            # Record times
            step_times["zero_grad"].append(zero_grad_time - step_start)
            step_times["forward"].append(forward_time - zero_grad_time)
            step_times["loss"].append(loss_time - forward_time)
            step_times["backward"].append(backward_time - loss_time)
            step_times["sync_gradients"].append(sync_time - backward_time)
            step_times["optimizer_step"].append(optimizer_time - sync_time)
            step_times["total"].append(optimizer_time - step_start)
            
            if rank == 0:
                print(f"Step {step + 1}/{NUM_STEPS}, Loss: {loss.item():.4f}, "
                      f"Time: {(optimizer_time - step_start) * 1000:.2f}ms")
    
    # Print average timing statistics (rank 0 only)
    if rank == 0:
        print("\n--- Average Step Timings (ms) ---")
        for name, times in step_times.items():
            avg_ms = sum(times) / len(times) * 1000
            print(f"  {name}: {avg_ms:.2f}ms")
        print("---------------------------------\n")
    
    # Save checkpoint from rank 0 only
    if rank == 0:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = CHECKPOINT_DIR / "final_ddp.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": NUM_STEPS,
            "loss": loss.item(),
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    cleanup()


def main():
    """Entry point: initialize model on CPU, spawn processes."""
    parser = argparse.ArgumentParser(description="Naive DDP training")
    parser.add_argument(
        "--no_distributed",
        action="store_true",
        help="Run on single GPU without distributed training",
    )
    args = parser.parse_args()
    
    world_size = torch.cuda.device_count()
    
    if world_size < 1:
        raise RuntimeError("No CUDA devices available")
    
    print("Initializing XL model on CPU...")
    torch.manual_seed(SEED)
    model = load_model("xl", device="cpu")
    print("Model initialized.")
    
    if args.no_distributed:
        train_single_gpu(model)
    else:
        print(f"Spawning {world_size} processes...")
        mp.spawn(
            train_ddp,
            args=(world_size, model),
            nprocs=world_size,
            join=True,
        )


if __name__ == "__main__":
    main()
