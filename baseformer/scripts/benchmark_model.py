"""
Simple benchmark script for model forward/backward pass.

Usage:
    uv run python -m baseformer.scripts.benchmark_model --model medium --steps 100 --warmup 10

    # With NVTX for Nsight profiling
    uv run nsys profile --pytorch=autograd-nvtx -t cuda,nvtx -o model_profile python -m baseformer.scripts.benchmark_model --model medium --nvtx
"""

import argparse
import time
from contextlib import nullcontext

import torch

from baseformer.nn import attention
from baseformer.nn.attention import annotated_scaled_dot_product_attention
from baseformer.optim.adamw import AdamW, annotated_adamw_step
from baseformer.scripts.utils import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--trace_output", default="trace.json", help="Output path for chrome trace")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--compile_mode", default="default", 
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    parser.add_argument("--nvtx", action="store_true", 
                        help="Use NVTX-annotated attention for Nsight profiling")
    parser.add_argument("--memory_profile", action="store_true",
                        help="Enable CUDA memory profiling snapshot")
    parser.add_argument("--memory_output", default="memory_snapshot.pickle",
                        help="Output path for memory snapshot")
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high')
    model = load_model(args.model, args.device)
    optimizer = AdamW(model.parameters(), lr=1e-4)

    # Swap in annotated functions at module level for NVTX profiling
    if args.nvtx:
        attention.scaled_dot_product_attention = annotated_scaled_dot_product_attention
        optimizer.step = lambda lr_schedule=None: annotated_adamw_step(optimizer, lr_schedule)

        print("Using NVTX-annotated scaled_dot_product_attention")
        print("Using NVTX-annotated adamw_step")



    if args.compile:
        print(f"Compiling model with mode={args.compile_mode}...")
        model = torch.compile(model, mode=args.compile_mode)

    # Random batch
    input_ids = torch.randint(0, 10048, (args.batch_size, args.seq_len), device=args.device)

    # Warmup
    for _ in range(args.warmup):
        output = model(input_ids)
        loss = output.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    print(f"Warmup complete")

    # Start recording memory history
    if args.memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    
    # Benchmark steps
    forward_times = []
    backward_times = []
    optimizer_times = []

    prof_context = (
        torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            profile_memory=True,
        )
        if args.profile
        else nullcontext()
    )

    with prof_context as prof:
        for step in range(args.steps):
            start = time.perf_counter()
            output = model(input_ids)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

            loss = output.sum()

            start = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - start)

            start = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize()
            optimizer_times.append(time.perf_counter() - start)

            optimizer.zero_grad()

    # Stop recording and dump memory snapshot
    if args.memory_profile:
        torch.cuda.memory._dump_snapshot(args.memory_output)
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f"Memory snapshot saved to: {args.memory_output}")

    # Report timings
    avg_forward = sum(forward_times) / len(forward_times) * 1000
    avg_backward = sum(backward_times) / len(backward_times) * 1000
    avg_optimizer = sum(optimizer_times) / len(optimizer_times) * 1000

    compile_str = f", compile={args.compile_mode}" if args.compile else ""
    print(f"Model: {args.model}, Batch: {args.batch_size}, Seq: {args.seq_len}{compile_str}")
    print(f"Forward:   {avg_forward:.2f} ms (avg over {args.steps} steps)")
    print(f"Backward:  {avg_backward:.2f} ms (avg over {args.steps} steps)")
    print(f"Optimizer: {avg_optimizer:.2f} ms (avg over {args.steps} steps)")
    print(f"Total:     {avg_forward + avg_backward + avg_optimizer:.2f} ms")

    # Profiler output
    if args.profile:
        prof.export_chrome_trace(args.trace_output)
        print(f"Chrome trace saved to: {args.trace_output}")


if __name__ == "__main__":
    main()
