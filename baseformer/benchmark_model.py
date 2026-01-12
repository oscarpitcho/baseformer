"""
Simple benchmark script for model forward/backward pass.

Usage:
    python -m baseformer.benchmark_model --model medium --steps 100 --warmup 10
"""

import argparse
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from omegaconf import OmegaConf

from baseformer.nn.transformer import TransformerLM
from baseformer.nn.position import RotaryPositionalEmbedding


VOCAB_SIZE = 10048

def load_model(model_name: str, device: str = "cuda") -> TransformerLM:
    """Load model from config files."""
    config_dir = Path(__file__).parent / "configs"

    base_cfg = OmegaConf.load(config_dir / "config.yaml")
    model_cfg = OmegaConf.load(config_dir / "model" / f"{model_name}.yaml")
    position_cfg = OmegaConf.load(config_dir / "position" / "rope.yaml")

    cfg = OmegaConf.merge(
        base_cfg,
        {"model": model_cfg},
        {"position": position_cfg},
        {"model": {"vocab_size": VOCAB_SIZE}},
    )

    d_k = cfg.model.d_model // cfg.model.num_heads
    rope = RotaryPositionalEmbedding(
        theta=cfg.position.theta,
        d_k=d_k,
        max_seq_len=cfg.position.max_seq_len,
        device=device,
    )

    model = TransformerLM(
        vocab_size=cfg.model.vocab_size,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
        d_ff=cfg.model.d_ff,
        rope=rope,
        device=device,
    )
    return model.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="medium")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profile", action="store_true", help="Enable PyTorch profiler")
    parser.add_argument("--trace_output", default="trace.json", help="Output path for chrome trace")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile()")
    parser.add_argument("--compile_mode", default="default", 
                        choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode")
    args = parser.parse_args()

    model = load_model(args.model, args.device)

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
        model.zero_grad()
    torch.cuda.synchronize()

    print(f"Warmup complete")


    
    # Benchmark steps
    forward_times = []
    backward_times = []

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

            model.zero_grad()

    # Report timings
    avg_forward = sum(forward_times) / len(forward_times) * 1000
    avg_backward = sum(backward_times) / len(backward_times) * 1000

    compile_str = f", compile={args.compile_mode}" if args.compile else ""
    print(f"Model: {args.model}, Batch: {args.batch_size}, Seq: {args.seq_len}{compile_str}")
    print(f"Forward:  {avg_forward:.2f} ms (avg over {args.steps} steps)")
    print(f"Backward: {avg_backward:.2f} ms (avg over {args.steps} steps)")
    print(f"Total:    {avg_forward + avg_backward:.2f} ms")

    # Profiler output
    if args.profile:
        prof.export_chrome_trace(args.trace_output)
        print(f"Chrome trace saved to: {args.trace_output}")


if __name__ == "__main__":
    main()
