import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from handover import HandoverManager
from lora_gemma_model import GemmaLoRAModel

print("=== [Step 2: Colab T4 GPU] LoRA Fine-Tuning Gemma Model ===")

input_bundle = "/content/handover_lora_1_to_2.tar.gz"
ho_in = HandoverManager.unpack(input_bundle)

x_train = ho_in.tensors["x_train"]
y_train = ho_in.tensors["y_train"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing LoRA Fine-Tuning on Device: {device}")

# Initialize Gemma LoRA Model
model = GemmaLoRAModel(vocab_size=32000, hidden_dim=768, num_layers=4, num_heads=12, r=8).to(device)

# Freeze base parameters, unfreeze ONLY LoRA parameters
model.mark_only_lora_as_trainable()

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Run single-burst training (3 Epochs)
model.train()
epochs = 3
t0_train = time.perf_counter()

for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
        
    avg_loss = total_loss / len(x_train)
    print(f"  Epoch [{epoch}/{epochs}] LoRA Training Loss: {avg_loss:.4f}")

train_time = time.perf_counter() - t0_train
print(f"⏱️ Colab GPU Single-Burst LoRA Training Completed in {train_time:.2f}s")

# Extract ONLY lightweight LoRA adapter state dict
lora_state_dict = model.get_lora_state_dict()

# Pack LoRA Adapter weights into handover bundle
ho_out = HandoverManager("lora_adapter_remote")
ho_out.add_meta("device_used", device)
ho_out.add_meta("final_train_loss", float(avg_loss))
ho_out.add_meta("epochs", epochs)
ho_out.add_meta("train_time_sec", float(train_time))

ho_out.tensors = lora_state_dict

output_bundle = "/content/handover_lora_adapter.tar.gz"
ho_out.pack(output_bundle)

print(f"✅ Step 2 complete. LoRA Adapter packed: {output_bundle}\n")
