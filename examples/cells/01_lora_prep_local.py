import os
import json
import torch
from handover import HandoverManager

print("=== [Step 1: Local CPU] Preprocessing & Compressing Instruction Dataset ===")

# Simulate 1,000 instruction-tuning text samples
num_samples = 1000
seq_len = 64
vocab_size = 32000

# Create raw synthetic JSON text dataset representation to measure raw file size
raw_instructions = []
for idx in range(num_samples):
    raw_instructions.append({
        "id": idx,
        "instruction": f"Explain AI topic #{idx} in detail with examples and steps.",
        "input": f"Context details for instruction #{idx} regarding machine learning tuning.",
        "output": f"Comprehensive answer for topic #{idx} with step-by-step guidance." * 5
    })

os.makedirs("./_tmp_raw_data", exist_ok=True)
raw_json_path = "./_tmp_raw_data/raw_dataset.json"
with open(raw_json_path, "w") as f:
    json.dump(raw_instructions, f, indent=2)

raw_size_bytes = os.path.getsize(raw_json_path)

# Tokenize / Convert to Compact PyTorch Tensors on Local CPU
x_train = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)
y_train = torch.randint(0, vocab_size, (num_samples, seq_len), dtype=torch.long)

# Pack into Handover Manager
ho = HandoverManager("lora_prep_local")
ho.add_meta("dataset_name", "Gemma-Instruction-Synthetic-1K")
ho.add_meta("num_samples", num_samples)
ho.add_meta("seq_len", seq_len)
ho.add_meta("raw_json_bytes", raw_size_bytes)

ho.add_tensor("x_train", x_train)
ho.add_tensor("y_train", y_train)

bundle_path = "handover_lora_1_to_2.tar.gz"
checksum = ho.pack(bundle_path)

bundle_size_bytes = os.path.getsize(bundle_path)
compression_ratio = (1 - (bundle_size_bytes / raw_size_bytes)) * 100

print(f"\n📊 --- Data Compression Metric (Local CPU) ---")
print(f"  - Raw Text JSON File Size:    {raw_size_bytes / 1e6:.2f} MB ({raw_size_bytes:,} bytes)")
print(f"  - Compressed Tensor Bundle:  {bundle_size_bytes / 1e6:.2f} MB ({bundle_size_bytes:,} bytes)")
print(f"  - Compression Ratio:          {compression_ratio:.1f}% Reduction! 🚀")
print(f"✅ Step 1 complete. Compact dataset packed: {bundle_path}\n")
