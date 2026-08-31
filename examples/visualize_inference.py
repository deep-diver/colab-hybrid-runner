import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from handover import HandoverManager
from mnist_model import SimpleMNISTCNN

print("=== Generating Visual Inference Plot ===")

bundle_file = "handover_mnist_3_to_4.tar.gz"
ho_in = HandoverManager.unpack(bundle_file)

x_sample = ho_in.tensors.pop("x_sample")
y_sample = ho_in.tensors.pop("y_sample")
state_dict = ho_in.tensors

# Load trained model
model = SimpleMNISTCNN()
model.load_state_dict(state_dict)
model.eval()

with torch.no_grad():
    outputs = model(x_sample)
    _, predictions = torch.max(outputs, 1)

# Plot 10 samples (2 rows, 5 columns)
fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Local Inference Test Results (Colab GPU Trained Model)", fontsize=16, fontweight='bold', y=0.98)

for i in range(10):
    ax = axes[i // 5, i % 5]
    img = x_sample[i].squeeze().numpy()
    true_label = y_sample[i].item()
    pred_label = predictions[i].item()
    
    ax.imshow(img, cmap='gray')
    ax.axis('off')
    
    is_match = (true_label == pred_label)
    title_color = 'green' if is_match else 'red'
    status_text = "MATCH" if is_match else "MISMATCH"
    
    ax.set_title(f"Sample #{i+1}\nTrue: {true_label} | Pred: {pred_label}\n[{status_text}]",
                 fontsize=11, fontweight='bold', color=title_color)

plt.tight_layout()

# Save image to artifact directory and workspace
artifact_dir = "/Users/chansungpark/.gemini/antigravity/brain/a8f462f1-a7a7-4fab-af48-eaebee241be5"
artifact_img_path = os.path.join(artifact_dir, "mnist_inference_results.png")
local_img_path = "./mnist_inference_results.png"

plt.savefig(artifact_img_path, dpi=150, bbox_inches='tight')
plt.savefig(local_img_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Visualization saved to:\n  - {artifact_img_path}\n  - {local_img_path}")
