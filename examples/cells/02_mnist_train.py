import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from handover import HandoverManager
from mnist_model import SimpleMNISTCNN

print("=== [Step 2: Colab Remote GPU] Training MNIST CNN Model ===")

# Unpack input data from Cell 1
input_bundle = "/content/handover_mnist_1_to_2.tar.gz"
ho_in = HandoverManager.unpack(input_bundle)

x_train = ho_in.tensors["x_train"]
y_train = ho_in.tensors["y_train"]
x_test = ho_in.tensors["x_test"]
y_test = ho_in.tensors["y_test"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing Training on Device: {device}")

model = SimpleMNISTCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Training loop (3 Epochs)
model.train()
epochs = 3
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)
    
    avg_loss = total_loss / len(x_train)
    print(f"Epoch [{epoch}/{epochs}] Average Training Loss: {avg_loss:.4f}")

# Pack trained model & test data for Step 3 Evaluation on Remote
ho_out = HandoverManager("mnist_train_remote")
ho_out.add_meta("device_used", device)
ho_out.add_meta("final_train_loss", float(avg_loss))
ho_out.add_meta("epochs", epochs)

# Save model state_dict
state_dict_tensors = {k: v.cpu() for k, v in model.state_dict().items()}
ho_out.tensors = state_dict_tensors

# Pass test data along to evaluation stage
ho_out.add_tensor("x_test", x_test)
ho_out.add_tensor("y_test", y_test)

output_bundle = "/content/handover_mnist_2_to_3.tar.gz"
ho_out.pack(output_bundle)

print(f"✅ Step 2 complete. Model trained on GPU and packed: {output_bundle}\n")
