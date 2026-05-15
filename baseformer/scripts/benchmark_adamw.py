"""
Benchmark script for AdamW optimizer: yours vs PyTorch's.

Usage:
    uv run python -m baseformer.scripts.benchmark_adamw --model small --steps 20

    # With NVTX for Nsight profiling
    uv run nsys profile --pytorch=autograd-nvtx -t cuda,nvtx -o adamw_profile python -m baseformer.scripts.benchmark_adamw --model small --nvtx
"""

import argparse
import time

import torch

from baseformer.scripts.utils import load_model
from baseformer.optim.adamw import AdamW as MyAdamW, annotated_adamw_step


def benchmark_optimizer(model, optimizer, input_ids, warmup, steps):
    """Run forward+backward+step and time just the optimizer step."""
    # Warmup
    for _ in range(warmup):
        loss = model(input_ids).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(steps):
        loss = model(input_ids).sum()
        loss.backward()
        torch.cuda.synchronize()

        start = time.perf_counter()
        optimizer.step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

        optimizer.zero_grad()

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX annotations for your AdamW")
    args = parser.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")

    input_ids = torch.randint(0, 10048, (args.batch_size, args.seq_len), device=device)

    # Benchmark your AdamW
    print(f"Benchmarking MyAdamW (nvtx={args.nvtx})...")
    model = load_model(args.model, device)
    my_optimizer = MyAdamW(model.parameters(), lr=1e-4)

    if args.nvtx:
        my_optimizer.step = lambda lr_schedule=None: annotated_adamw_step(my_optimizer, lr_schedule)

    my_times = benchmark_optimizer(model, my_optimizer, input_ids, args.warmup, args.steps)

    del model, my_optimizer
    torch.cuda.empty_cache()

    # Benchmark torch AdamW
    print("Benchmarking torch.optim.AdamW...")
    model = load_model(args.model, device)
    torch_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    torch_times = benchmark_optimizer(model, torch_optimizer, input_ids, args.warmup, args.steps)

    del model, torch_optimizer
    torch.cuda.empty_cache()

    # Report
    my_avg = sum(my_times) / len(my_times) * 1000
    torch_avg = sum(torch_times) / len(torch_times) * 1000

    print()
    print(f"Model: {args.model}, Batch: {args.batch_size}, Seq: {args.seq_len}")
    print(f"MyAdamW:    {my_avg:.2f} ms")
    print(f"TorchAdamW: {torch_avg:.2f} ms")
    print(f"Ratio:      {my_avg / torch_avg:.2f}x")


if __name__ == "__main__":
    main()
