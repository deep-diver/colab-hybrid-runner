import time
import torch
import torch.nn as nn

# Self-contained 4-layer Transformer Language Model Architecture (Llama/Gemma style)
class TransformerLM(nn.Module):
    def __init__(self, vocab_size=32000, hidden_dim=768, num_layers=4, num_heads=12):
        super(TransformerLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.transformer(x)
        logits = self.lm_head(x)
        return logits

def run_end_to_end_lm_benchmark():
    print("=================================================================")
    print("🚀 END-TO-END AUTOREGRESSIVE LLM GENERATION BENCHMARK")
    print("   Comparing Mode A (GPU-Only) vs Mode B (CPU-GPU Hybrid)")
    print("=================================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device Name: {torch.cuda.get_device_name(0)}")

    vocab_size = 32000
    hidden_dim = 768
    model = TransformerLM(vocab_size=vocab_size, hidden_dim=hidden_dim).to(device)
    model.eval()

    # Input Prompt (8 tokens)
    prompt_ids = torch.randint(0, vocab_size, (1, 8))
    max_gen_tokens = 50

    # -------------------------------------------------------------------------
    # MODE A: GPU-Only Text Generation (Forward + Logits + Sampling ALL on GPU)
    # -------------------------------------------------------------------------
    print("\n--- Running MODE A: GPU-Only Autoregressive Generation ---")
    gpu_input_ids = prompt_ids.to(device)
    gen_tokens_gpu = []

    torch.cuda.synchronize()
    t0_gpu = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_gen_tokens):
            # 1. Forward Pass on GPU
            logits = model(gpu_input_ids)[:, -1, :] # (1, 32000)
            
            # 2. Temperature + Top-K (K=50) Sampling on GPU
            logits = logits / 0.7
            v_gpu, i_gpu = torch.topk(logits, 50)
            probs = torch.softmax(v_gpu, dim=-1)
            next_token = i_gpu[0, torch.multinomial(probs, 1)]
            
            gen_tokens_gpu.append(next_token.item())
            gpu_input_ids = torch.cat([gpu_input_ids, next_token.unsqueeze(0)], dim=1)

    torch.cuda.synchronize()
    gpu_total_time = time.perf_counter() - t0_gpu
    gpu_tps = len(gen_tokens_gpu) / gpu_total_time

    print(f"GPU-Only Generation Time: {gpu_total_time:.4f} s ({gpu_tps:.2f} tokens/s)")

    # -------------------------------------------------------------------------
    # MODE B: CPU-GPU Hybrid Text Generation (Forward on GPU + Sampling on CPU)
    # -------------------------------------------------------------------------
    print("\n--- Running MODE B: CPU-GPU Hybrid Autoregressive Generation ---")
    hybrid_input_ids = prompt_ids.to(device)
    gen_tokens_hybrid = []

    torch.cuda.synchronize()
    t0_hybrid = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_gen_tokens):
            # 1. Forward Pass on GPU
            logits_gpu = model(hybrid_input_ids)[:, -1, :] # (1, 32000)
            
            # 2. Transfer Logits to CPU & Top-K Sampling on CPU
            logits_cpu = logits_gpu.to("cpu", non_blocking=True) / 0.7
            v_cpu, i_cpu = torch.topk(logits_cpu, 50)
            probs_cpu = torch.softmax(v_cpu, dim=-1)
            next_token_cpu = i_cpu[0, torch.multinomial(probs_cpu, 1)]
            
            next_token_val = next_token_cpu.item()
            gen_tokens_hybrid.append(next_token_val)
            
            # Send next token back to GPU input sequence
            next_token_gpu_tensor = torch.tensor([[next_token_val]], device=device)
            hybrid_input_ids = torch.cat([hybrid_input_ids, next_token_gpu_tensor], dim=1)

    torch.cuda.synchronize()
    hybrid_total_time = time.perf_counter() - t0_hybrid
    hybrid_tps = len(gen_tokens_hybrid) / hybrid_total_time

    print(f"CPU-GPU Hybrid Generation Time: {hybrid_total_time:.4f} s ({hybrid_tps:.2f} tokens/s)")

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON
    # -------------------------------------------------------------------------
    print("\n=================================================================")
    print("📊 REAL END-TO-END AUTOREGRESSIVE GENERATION SUMMARY")
    print("=================================================================")
    print(f"MODE A (GPU-Only):    {gpu_total_time:.4f} s ({gpu_tps:.2f} tokens/sec)")
    print(f"MODE B (CPU-Hybrid):  {hybrid_total_time:.4f} s ({hybrid_tps:.2f} tokens/sec)")
    speedup = (gpu_total_time / hybrid_total_time)
    print(f"🚀 Generation Speedup Factor (Hybrid vs GPU-Only): {speedup:.2f}x")

if __name__ == "__main__":
    run_end_to_end_lm_benchmark()
