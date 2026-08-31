import time
import torch
import torch.nn as nn

def run_gemma_benchmark():
    print("===============================================================")
    print("🔬 EMPIRICAL BENCHMARK: Gemma Model Architecture (Vocab=262,144)")
    print("   Comparing GPU-Only vs CPU-GPU Hybrid Execution")
    print("===============================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Initial VRAM Allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")

    # =========================================================================
    # TEST 1: Gemma Embedding Layer (Vocab = 262,144, Hidden Dim = 3,584)
    # =========================================================================
    print("\n---------------------------------------------------------------")
    print("TEST 1: Gemma Embedding Layer (Vocab=262,144, Dim=3,584)")
    print("---------------------------------------------------------------")
    
    vocab_size = 262144
    hidden_dim = 3584
    batch_tokens = torch.randint(0, vocab_size, (1, 128)) # 128 tokens

    # 1-A: GPU-Only Embedding
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated()
    emb_gpu = nn.Embedding(vocab_size, hidden_dim).to(device)
    vram_after_gpu_emb = torch.cuda.memory_allocated()
    vram_cost_mb = (vram_after_gpu_emb - vram_before) / 1e6

    tokens_gpu = batch_tokens.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        _ = emb_gpu(tokens_gpu)
    torch.cuda.synchronize()
    time_gpu_emb = (time.perf_counter() - t0) / 200 * 1000  # ms

    # 1-B: CPU-GPU Hybrid Embedding (Offloaded to CPU RAM)
    emb_cpu = nn.Embedding(vocab_size, hidden_dim).to("cpu")
    tokens_cpu = batch_tokens.to("cpu")
    
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        vec_cpu = emb_cpu(tokens_cpu)
        vec_gpu = vec_cpu.to(device, non_blocking=True) # Transfer lookup result (128x3584) to GPU
    torch.cuda.synchronize()
    time_hybrid_emb = (time.perf_counter() - t0) / 200 * 1000  # ms

    print(f"GPU-Only Embedding VRAM Cost:  {vram_cost_mb:.2f} MB ({vram_cost_mb/1024:.2f} GB)")
    print(f"GPU-Only Lookup Latency:       {time_gpu_emb:.4f} ms")
    print(f"CPU-GPU Hybrid Lookup Latency: {time_hybrid_emb:.4f} ms")
    print(f"💡 VRAM Saved by Offloading Gemma Embedding to CPU: {vram_cost_mb:.2f} MB ({vram_cost_mb/1024:.2f} GB)")

    del emb_gpu, tokens_gpu
    torch.cuda.empty_cache()

    # =========================================================================
    # TEST 2: Gemma Logit Sampling (Batch Size = 1, Vocab Size = 262,144)
    # =========================================================================
    print("\n---------------------------------------------------------------")
    print("TEST 2: Gemma Token Logit Sampling (Batch=1, Vocab=262,144)")
    print("---------------------------------------------------------------")
    
    logits_raw = torch.randn(1, vocab_size)

    # 2-A: GPU-Only Sampling
    logits_gpu = logits_raw.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        l_gpu = logits_gpu / 0.7
        v_gpu, i_gpu = torch.topk(l_gpu, 50)
        probs_gpu = torch.softmax(v_gpu, dim=-1)
        next_token_gpu = i_gpu[0, torch.multinomial(probs_gpu, 1)]
    torch.cuda.synchronize()
    time_gpu_sample = (time.perf_counter() - t0) / 200 * 1000 # ms

    # 2-B: CPU-GPU Hybrid Sampling
    logits_gpu_out = logits_raw.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        l_cpu = logits_gpu_out.to("cpu", non_blocking=True)
        l_cpu = l_cpu / 0.7
        v_cpu, i_cpu = torch.topk(l_cpu, 50)
        probs_cpu = torch.softmax(v_cpu, dim=-1)
        next_token_cpu = i_cpu[0, torch.multinomial(probs_cpu, 1)]
    torch.cuda.synchronize()
    time_hybrid_sample = (time.perf_counter() - t0) / 200 * 1000 # ms

    print(f"GPU-Only Sampling Latency:       {time_gpu_sample:.4f} ms per token")
    print(f"CPU-GPU Hybrid Sampling Latency: {time_hybrid_sample:.4f} ms per token")
    speedup_sample = (time_gpu_sample / time_hybrid_sample) if time_hybrid_sample > 0 else 0
    print(f"💡 Gemma Sampling Speedup Ratio (CPU vs GPU): {speedup_sample:.2f}x")

    print("\n===============================================================")
    print("📊 GEMMA BENCHMARK SUMMARY")
    print("===============================================================")
    print(f"1. Embedding Layer: CPU Offloading saves {vram_cost_mb/1024:.2f} GB VRAM.")
    print(f"2. Token Sampling:  CPU Sampling is {speedup_sample:.2f}x faster for Gemma 262k Vocab.")

if __name__ == "__main__":
    run_gemma_benchmark()
