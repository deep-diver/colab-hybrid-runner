import sys
import subprocess

try:
    import transformers
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "accelerate"])

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("=================================================================")
    print("🚀 END-TO-END LLM TEXT GENERATION BENCHMARK (GPU vs Hybrid)")
    print("=================================================================")

    prompt = "Explain quantum computing in 3 simple bullet points."
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"  # Fast, high quality open-weights LLM
    print(f"Loading Model & Tokenizer: {model_id}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device} ({torch.cuda.get_device_name(0)})")

    # -------------------------------------------------------------------------
    # STEP 1: Tokenization (Local CPU)
    # -------------------------------------------------------------------------
    t0_tok = time.perf_counter()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    tok_time = (time.perf_counter() - t0_tok) * 1000
    print(f"\n[Step 1] Tokenization (CPU): {tok_time:.2f} ms | Input Tokens: {input_ids.shape[1]}")

    # -------------------------------------------------------------------------
    # MODE A: GPU-Only Text Generation (Transformer + Logits + Sampling on GPU)
    # -------------------------------------------------------------------------
    print("\n--- Running MODE A: GPU-Only Generation ---")
    gen_tokens_gpu = []
    curr_input_ids = input_ids.to(device)
    
    max_new_tokens = 50
    torch.cuda.synchronize()
    t0_gpu = time.perf_counter()
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. Forward Pass on GPU
            outputs = model(curr_input_ids)
            next_token_logits = outputs.logits[:, -1, :] # (1, vocab_size)
            
            # 2. Sampling on GPU
            next_token_logits = next_token_logits / 0.7
            filtered_logits, filtered_indices = torch.topk(next_token_logits, 50)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token = filtered_indices[0, torch.multinomial(probs, 1)]
            
            gen_tokens_gpu.append(next_token.item())
            curr_input_ids = torch.cat([curr_input_ids, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    gpu_total_time = time.perf_counter() - t0_gpu
    gpu_tps = len(gen_tokens_gpu) / gpu_total_time

    text_gpu = tokenizer.decode(gen_tokens_gpu, skip_special_tokens=True)
    print(f"GPU-Only Generation Time: {gpu_total_time:.2f} s ({gpu_tps:.2f} tokens/s)")
    print(f"Generated Text:\n{text_gpu}")

    # -------------------------------------------------------------------------
    # MODE B: CPU-GPU Hybrid Text Generation (Transformer on GPU + Sampling on CPU)
    # -------------------------------------------------------------------------
    print("\n--- Running MODE B: CPU-GPU Hybrid Generation ---")
    gen_tokens_hybrid = []
    curr_input_ids_h = input_ids.to(device)
    
    torch.cuda.synchronize()
    t0_hybrid = time.perf_counter()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # 1. Forward Pass on GPU
            outputs = model(curr_input_ids_h)
            next_token_logits_gpu = outputs.logits[:, -1, :]
            
            # 2. Transfer Logits to CPU & Sample on CPU
            logits_cpu = next_token_logits_gpu.to("cpu", non_blocking=True) / 0.7
            filtered_logits, filtered_indices = torch.topk(logits_cpu, 50)
            probs = torch.softmax(filtered_logits, dim=-1)
            next_token_cpu = filtered_indices[0, torch.multinomial(probs, 1)]
            
            next_token_val = next_token_cpu.item()
            gen_tokens_hybrid.append(next_token_val)
            
            # Send chosen token ID back to GPU input sequence
            next_token_gpu_tensor = torch.tensor([[next_token_val]], device=device)
            curr_input_ids_h = torch.cat([curr_input_ids_h, next_token_gpu_tensor], dim=1)
            
            if next_token_val == tokenizer.eos_token_id:
                break

    torch.cuda.synchronize()
    hybrid_total_time = time.perf_counter() - t0_hybrid
    hybrid_tps = len(gen_tokens_hybrid) / hybrid_total_time

    text_hybrid = tokenizer.decode(gen_tokens_hybrid, skip_special_tokens=True)
    print(f"CPU-GPU Hybrid Generation Time: {hybrid_total_time:.2f} s ({hybrid_tps:.2f} tokens/s)")
    print(f"Generated Text:\n{text_hybrid}")

    # -------------------------------------------------------------------------
    # SUMMARY COMPARISON
    # -------------------------------------------------------------------------
    print("\n=================================================================")
    print("📊 REAL END-TO-END GENERATION BENCHMARK SUMMARY")
    print("=================================================================")
    print(f"MODE A (GPU-Only):    {gpu_total_time:.2f} s ({gpu_tps:.2f} tokens/sec)")
    print(f"MODE B (CPU-Hybrid):  {hybrid_total_time:.2f} s ({hybrid_tps:.2f} tokens/sec)")
    speedup = (gpu_total_time / hybrid_total_time)
    print(f"🚀 Speedup Factor (Hybrid vs GPU-Only): {speedup:.2f}x")

if __name__ == "__main__":
    main()
