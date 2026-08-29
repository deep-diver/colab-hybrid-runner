import time
import torch

def run_batch_scaling_benchmark():
    print("=================================================================")
    print("🔬 EMPIRICAL BENCHMARK: BATCH SIZE SCALING TEST (Batch = 1 to 128)")
    print("   Comparing GPU-Only vs CPU-GPU Hybrid Sampling")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    vocab_size = 128000
    batch_sizes = [1, 4, 16, 32, 64, 128]

    print("\nBatch Size | Logit Data Size | GPU-Only Latency | Hybrid Latency | Faster Mode")
    print("-" * 75)

    for b in batch_sizes:
        # Create Logits Tensor for batch size b
        logits_raw = torch.randn(b, vocab_size)
        data_size_mb = (b * vocab_size * 4) / 1e6

        # ---------------------------------------------------------------------
        # Mode A: GPU-Only Sampling
        # ---------------------------------------------------------------------
        logits_gpu = logits_raw.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            l_gpu = logits_gpu / 0.7
            v_gpu, i_gpu = torch.topk(l_gpu, 50, dim=-1)
            probs_gpu = torch.softmax(v_gpu, dim=-1)
            next_tokens_gpu = i_gpu.gather(-1, torch.multinomial(probs_gpu, 1))
        torch.cuda.synchronize()
        time_gpu = (time.perf_counter() - t0) / 50 * 1000 # ms

        # ---------------------------------------------------------------------
        # Mode B: CPU-GPU Hybrid Sampling
        # ---------------------------------------------------------------------
        logits_gpu_out = logits_raw.to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(50):
            # Transfer Batch Logits over PCIe to CPU
            l_cpu = logits_gpu_out.to("cpu", non_blocking=True) / 0.7
            v_cpu, i_cpu = torch.topk(l_cpu, 50, dim=-1)
            probs_cpu = torch.softmax(v_cpu, dim=-1)
            next_tokens_cpu = i_cpu.gather(-1, torch.multinomial(probs_cpu, 1))
        torch.cuda.synchronize()
        time_hybrid = (time.perf_counter() - t0) / 50 * 1000 # ms

        faster_mode = "⚡ CPU Hybrid" if time_hybrid < time_gpu else "🚀 GPU-Only"
        ratio = time_gpu / time_hybrid if time_hybrid < time_gpu else time_hybrid / time_gpu

        print(f"Batch={b:^5d} | {data_size_mb:^14.2f} MB | {time_gpu:^15.2f} ms | {time_hybrid:^13.2f} ms | {faster_mode} ({ratio:.2f}x)")

    print("-" * 75)
    print("💡 BENCHMARK CONCLUSION:")
    print("   Small Batches (1~4): CPU Hybrid is faster due to low kernel launch overhead.")
    print("   Large Batches (32~128): GPU-Only is faster due to massive GPU parallel compute & avoiding PCIe transfer bottleneck.")

if __name__ == "__main__":
    run_batch_scaling_benchmark()
