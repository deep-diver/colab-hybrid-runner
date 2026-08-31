import torch
from handover import HandoverManager
from mnist_model import SimpleMNISTCNN

print("=== [Step 4: Local CPU] Running Inference Test with Trained Model ===")

bundle_file = "handover_mnist_3_to_4.tar.gz"
ho_in = HandoverManager.unpack(bundle_file)

meta = ho_in.metadata
x_sample = ho_in.tensors.pop("x_sample")
y_sample = ho_in.tensors.pop("y_sample")

state_dict = ho_in.tensors

print(f"Remote Evaluation Accuracy: {meta.get('test_accuracy'):.2f}%")

# Load model locally on CPU
model = SimpleMNISTCNN()
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    outputs = model(x_sample)
    _, predictions = torch.max(outputs, 1)

print("\n--- Local Inference Results (Sample Test Images) ---")
print("Sample Index | Ground Truth Label | Model Prediction | Match Status")
print("-" * 65)

matches = 0
for idx in range(len(y_sample)):
    true_label = y_sample[idx].item()
    pred_label = predictions[idx].item()
    is_correct = "✅ MATCH" if true_label == pred_label else "❌ MISMATCH"
    if true_label == pred_label:
        matches += 1
    print(f"Sample #{idx+1:02d}    | {true_label:^18d} | {pred_label:^16d} | {is_correct}")

print("-" * 65)
print(f"Local Inference Sample Match Rate: {matches}/{len(y_sample)} ({matches/len(y_sample)*100:.1f}%)")
print("\n🎉 Full MNIST Hybrid Pipeline (Prep -> Train -> Eval -> Local Inference) Completed Successfully!")
