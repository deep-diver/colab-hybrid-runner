import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from handover import HandoverManager
from mnist_model import SimpleMNISTCNN

print("=== [Step 3: Colab Remote GPU] Evaluating MNIST Model ===")

input_bundle = "/content/handover_mnist_2_to_3.tar.gz"
ho_in = HandoverManager.unpack(input_bundle)

x_test = ho_in.tensors.pop("x_test")
y_test = ho_in.tensors.pop("y_test")

state_dict = ho_in.tensors

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing Evaluation on Device: {device}")

model = SimpleMNISTCNN().to(device)
model.load_state_dict(state_dict)
model.eval()

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

correct = 0
total = 0

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

accuracy = 100.0 * correct / total
print(f"🎯 Test Accuracy on Remote Colab GPU: {accuracy:.2f}% ({correct}/{total})")

ho_out = HandoverManager("mnist_eval_remote")
ho_out.add_meta("eval_device", device)
ho_out.add_meta("test_accuracy", float(accuracy))
ho_out.add_meta("correct", int(correct))
ho_out.add_meta("total", int(total))

# Include model state dict & sample test images for local inference
ho_out.tensors = state_dict
ho_out.add_tensor("x_sample", x_test[:10])
ho_out.add_tensor("y_sample", y_test[:10])

output_bundle = "/content/handover_mnist_3_to_4.tar.gz"
ho_out.pack(output_bundle)

print(f"✅ Step 3 complete. Evaluation finished on GPU and packed: {output_bundle}\n")
