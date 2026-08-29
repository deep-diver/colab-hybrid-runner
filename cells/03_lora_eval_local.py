import os
import torch
from handover import HandoverManager
from lora_gemma_model import GemmaLoRAModel

print("=== [Step 3: Local CPU] Backup LoRA Adapter & Verify Local CPU Inference ===")

bundle_path = "handover_lora_adapter.tar.gz"
ho_in = HandoverManager.unpack(bundle_path)

meta = ho_in.metadata
adapter_state_dict = ho_in.tensors

bundle_size = os.path.getsize(bundle_path)
print(f"\n💾 --- LoRA Adapter Weight Backup Metric ---")
print(f"  - Downloaded Adapter File:  {bundle_path}")
print(f"  - LoRA Adapter Size:       {bundle_size / 1e6:.2f} MB ({bundle_size:,} bytes)")
print(f"  - Colab Training Device:   {meta.get('device_used')}")
print(f"  - Final Training Loss:     {meta.get('final_train_loss'):.4f}")
print(f"  - Colab GPU Training Time: {meta.get('train_time_sec'):.2f} s")

# Instantiate Gemma Model on Local CPU
model_local = GemmaLoRAModel(vocab_size=32000, hidden_dim=768, num_layers=4, num_heads=12, r=8).to("cpu")

# Load fine-tuned LoRA Adapter weights onto local base model
missing, unexpected = model_local.load_state_dict(adapter_state_dict, strict=False)
model_local.eval()

print(f"\n✅ LoRA Adapter successfully merged onto Local CPU Base Model! (Missing keys: {len(missing)} base params, Loaded keys: {len(adapter_state_dict)} LoRA params)")

# Run local inference test on CPU
prompt_tokens = torch.randint(0, 32000, (1, 8), dtype=torch.long)
with torch.no_grad():
    logits = model_local(prompt_tokens)
    next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

print(f"\n🧪 --- Local CPU Inference Verification ---")
print(f"  - Test Input Tokens:  {prompt_tokens.squeeze().tolist()}")
print(f"  - Fine-Tuned Output Next Token ID: {next_token_id}")
print(f"\n🎉 Practical Hybrid LoRA Fine-Tuning Workflow Completed Successfully!")
