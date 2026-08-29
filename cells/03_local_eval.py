import sys
import pandas as pd
from handover import HandoverManager

print("=== [Step 3: Local] Evaluation & Final Report ===")

# Unpack bundle downloaded from Colab
bundle_file = "handover_2_to_3.tar.gz"
ho = HandoverManager.unpack(bundle_file)

meta = ho.metadata
df_results = ho.tables["processed_results"]
predictions = ho.tensors["predictions"]

print("\n--- Handover Execution Report ---")
print(f"Executed on Colab Device: {meta.get('device_used')}")
print(f"Status: {meta.get('status')}")
print(f"Predictions Tensor Shape: {predictions.shape}")
print("\nFinal Output DataFrame:")
print(df_results)

print("\n✅ Hybrid Workflow (Local -> Colab -> Local) Completed Successfully!")
