import os
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Data sizes in number of float32 elements
# float32 = 4 bytes, so: 1MB = 250,000 elements, etc.
DATA_SIZES = {
    "1MB": 250_000,
    "10MB": 2_500_000,
    "100MB": 25_000_000,
    "1GB": 250_000_000,
}

BACKENDS = ["gloo", "nccl"]
WORLD_SIZES = [2, 4, 6]


def setup(rank, world_size, backend):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def benchmark_allreduce(rank, world_size, backend, num_elements, result_dict, num_iterations=50, warmup=10):
    """
    Benchmark the all-reduce operation.
    
    Args:
        rank: Process rank
        world_size: Total number of processes
        backend: "gloo" or "nccl"
        num_elements: Number of float32 elements in the tensor
        result_dict: Shared dict to store results (only rank 0 writes)
        num_iterations: Number of timed iterations
        warmup: Number of warmup iterations
    """
    setup(rank, world_size, backend)
    
    # Create tensor on appropriate device
    device = f"cuda:{rank}" if backend == "nccl" else "cpu"
    data = torch.randn(num_elements, dtype=torch.float32, device=device)
    
    # Warmup iterations
    for _ in range(warmup):
        dist.all_reduce(data, async_op=False)
    
    # Synchronize before timing
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    
    # Timed iterations
    start = time.perf_counter()
    for _ in range(num_iterations):
        dist.all_reduce(data, async_op=False)
    if backend == "nccl":
        torch.cuda.synchronize()
    dist.barrier()
    end = time.perf_counter()
    
    # Only rank 0 reports results
    if rank == 0:
        avg_time_ms = (end - start) / num_iterations * 1000  # Convert to ms
        result_dict["avg_time_ms"] = avg_time_ms
    
    cleanup()


def run_benchmark(backend, world_size, size_name, num_elements):
    """
    Run a single benchmark configuration.
    
    Returns:
        Average time in milliseconds, or None if skipped.
    """
    # Check GPU availability for NCCL
    if backend == "nccl":
        num_gpus = torch.cuda.device_count()
        if num_gpus < world_size:
            print(f"  Skipping NCCL with {world_size} processes: only {num_gpus} GPUs available")
            return None
    
    # Use a manager dict to share results between processes
    manager = mp.Manager()
    result_dict = manager.dict()
    
    mp.spawn(
        fn=benchmark_allreduce,
        args=(world_size, backend, num_elements, result_dict),
        nprocs=world_size,
        join=True
    )
    
    return result_dict.get("avg_time_ms")


def main():
    results = []
    
    print("=" * 60)
    print("All-Reduce Benchmark")
    print("=" * 60)
    
    for backend in BACKENDS:
        device = "GPU" if backend == "nccl" else "CPU"
        print(f"\nBackend: {backend.upper()} ({device})")
        print("-" * 40)
        
        for world_size in WORLD_SIZES:
            print(f"\n  Processes: {world_size}")
            
            for size_name, num_elements in DATA_SIZES.items():
                avg_time = run_benchmark(backend, world_size, size_name, num_elements)
                
                if avg_time is not None:
                    print(f"    {size_name}: {avg_time:.3f} ms")
                    results.append({
                        "backend": backend,
                        "device": device,
                        "data_size": size_name,
                        "num_processes": world_size,
                        "avg_time_ms": avg_time
                    })
    
    # Save results to CSV
    csv_path = "benchmark_results.csv"
    with open(csv_path, "w") as f:
        f.write("backend,device,data_size,num_processes,avg_time_ms\n")
        for r in results:
            f.write(f"{r['backend']},{r['device']},{r['data_size']},{r['num_processes']},{r['avg_time_ms']:.3f}\n")
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to {csv_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
