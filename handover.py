import os
import json
import tarfile
import shutil
import hashlib
import torch
import pandas as pd
from typing import Dict, Optional, Tuple

def get_file_sha256(filepath: str) -> str:
    """Calculates SHA256 checksum of a file."""
    if not os.path.exists(filepath):
        return ""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

class HandoverPolicy:
    """Policy for Source of Truth (SoT) and Caching behavior per step."""
    LOCAL = "LOCAL"            # Local environment is Source of Truth
    REMOTE = "REMOTE"          # Remote (Colab VM) is Source of Truth
    AUTO_CACHE = "AUTO_CACHE"  # Compare SHA256 hash & skip transfer if unchanged

    def __init__(self, sot_map: Optional[Dict[str, str]] = None, enable_caching: bool = True):
        self.sot_map = sot_map or {}
        self.enable_caching = enable_caching

    def get_sot(self, step_name: str) -> str:
        return self.sot_map.get(step_name, self.AUTO_CACHE)

class HandoverManager:
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.metadata = {"step": step_name}
        self.tensors = {}
        self.tables = {}

    def add_meta(self, key, value):
        self.metadata[key] = value

    def add_tensor(self, key, tensor: torch.Tensor):
        self.tensors[key] = tensor

    def add_dataframe(self, key, df: pd.DataFrame):
        self.tables[key] = df

    def pack(self, output_path: str) -> str:
        tmp_dir = f"./_tmp_pack_{self.step_name}"
        data_dir = os.path.join(tmp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)

        if self.tensors:
            torch.save(self.tensors, os.path.join(data_dir, "tensors.pt"))

        for name, df in self.tables.items():
            df.to_parquet(os.path.join(data_dir, f"{name}.parquet"))

        with open(os.path.join(tmp_dir, "metadata.json"), "w") as f:
            json.dump(self.metadata, f, indent=2)

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(tmp_dir, arcname=".")

        shutil.rmtree(tmp_dir)
        checksum = get_file_sha256(output_path)
        print(f"📦 [HandoverManager] Packed: {output_path} (SHA256: {checksum[:8]}...)")
        return checksum

    @classmethod
    def unpack(cls, bundle_path: str):
        tmp_dir = f"./_tmp_unpack_{os.path.basename(bundle_path)}"
        os.makedirs(tmp_dir, exist_ok=True)

        with tarfile.open(bundle_path, "r:gz") as tar:
            tar.extractall(tmp_dir)

        with open(os.path.join(tmp_dir, "metadata.json"), "r") as f:
            meta = json.load(f)

        instance = cls(meta.get("step", "unknown"))
        instance.metadata = meta

        tensor_file = os.path.join(tmp_dir, "data", "tensors.pt")
        if os.path.exists(tensor_file):
            instance.tensors = torch.load(tensor_file, weights_only=True)

        data_dir = os.path.join(tmp_dir, "data")
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".parquet"):
                    key = file[:-8]
                    instance.tables[key] = pd.read_parquet(os.path.join(data_dir, file))

        shutil.rmtree(tmp_dir)
        print(f"📂 [HandoverManager] Unpacked: {bundle_path} (Step: {instance.step_name})")
        return instance
