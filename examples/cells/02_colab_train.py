import os
import torch
import pandas as pd
from handover import HandoverManager

print("=== [Step 2: Colab VM] Executing Heavy Computation / Training ===")

# Unpack input handover bundle from Cell 1
input_bundle = "/content/handover_1_to_2.tar.gz"
ho_in = HandoverManager.unpack(input_bundle)

# Extract data
meta = ho_in.metadata
df = ho_in.tables["user_features"]
weights = ho_in.tensors["initial_weights"]

print(f"Meta Received: {meta}")
print(f"DataFrame Received:\n{df}")
print(f"Initial Weights Shape: {weights.shape}")

# Convert DataFrame features to PyTorch Tensor for matrix multiplication
features = torch.tensor(df[["feature_a", "feature_b"]].values, dtype=torch.float32)

# Check if CUDA (GPU) is available on Colab
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Executing computation on device: {device}")

features = features.to(device)
weights = weights.to(device)

# Simulate GPU Matrix Computation / Forward Pass
predictions = torch.matmul(features, weights) * 2.5

print(f"Computed Predictions on Colab:\n{predictions}")

# Pack results into Step 2 -> Step 3 Handover Bundle
ho_out = HandoverManager("step2_colab_train")
ho_out.add_meta("device_used", device)
ho_out.add_meta("status", "SUCCESS")
ho_out.add_tensor("predictions", predictions.cpu())

# Append prediction column back to dataframe
df["prediction"] = predictions.cpu().squeeze().tolist()
ho_out.add_dataframe("processed_results", df)

output_bundle = "/content/handover_2_to_3.tar.gz"
ho_out.pack(output_bundle)

print(f"Cell 2 completed on Colab. Output bundle packed: {output_bundle}\n")
