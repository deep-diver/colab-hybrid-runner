import sys
import torch
import pandas as pd
from handover import HandoverManager

print("=== [Step 1: Local] Preparing Data & Hyperparameters ===")

# Create synthetic dataset
df = pd.DataFrame({
    "user_id": [101, 102, 103, 104, 105],
    "feature_a": [1.2, 3.4, 5.1, 2.8, 4.0],
    "feature_b": [10.0, 20.0, 15.0, 25.0, 30.0]
})

# Create weight matrix tensor
initial_weights = torch.tensor([[0.5], [1.5]], dtype=torch.float32)

# Pack into Handover Bundle
ho = HandoverManager("step1_local_prep")
ho.add_meta("author", "Antigravity Agent")
ho.add_meta("learning_rate", 0.01)
ho.add_meta("target_epochs", 5)
ho.add_dataframe("user_features", df)
ho.add_tensor("initial_weights", initial_weights)

bundle_filename = "handover_1_to_2.tar.gz"
ho.pack(bundle_filename)

print(f"Cell 1 completed locally. Created: {bundle_filename}\n")
