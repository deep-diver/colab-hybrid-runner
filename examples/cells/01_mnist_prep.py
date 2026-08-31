import os
import gzip
import numpy as np
import urllib.request
import torch
from handover import HandoverManager

print("=== [Step 1: Local] MNIST Data Download & Preprocessing ===")

def load_mnist_subset(num_train=2000, num_test=500):
    url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz",
    }
    
    data = {}
    os.makedirs("./_mnist_cache", exist_ok=True)
    
    for key, filename in files.items():
        local_path = os.path.join("./_mnist_cache", filename)
        if not os.path.exists(local_path):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url_base + filename, local_path)
            
    with gzip.open("./_mnist_cache/train-images-idx3-ubyte.gz", "rb") as f:
        x_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open("./_mnist_cache/train-labels-idx1-ubyte.gz", "rb") as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
        
    with gzip.open("./_mnist_cache/t10k-images-idx3-ubyte.gz", "rb") as f:
        x_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
    with gzip.open("./_mnist_cache/t10k-labels-idx1-ubyte.gz", "rb") as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        
    # Take subset & normalize to [0.0, 1.0]
    x_train = torch.tensor(x_train[:num_train], dtype=torch.float32) / 255.0
    y_train = torch.tensor(y_train[:num_train], dtype=torch.long)
    x_test = torch.tensor(x_test[:num_test], dtype=torch.float32) / 255.0
    y_test = torch.tensor(y_test[:num_test], dtype=torch.long)
    
    return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = load_mnist_subset()

print(f"Train dataset shape: {x_train.shape}, labels: {y_train.shape}")
print(f"Test dataset shape:  {x_test.shape}, labels: {y_test.shape}")

# Pack into Handover Manager
ho = HandoverManager("mnist_prep_local")
ho.add_meta("dataset", "MNIST")
ho.add_meta("num_train", len(x_train))
ho.add_meta("num_test", len(x_test))
ho.add_meta("image_shape", list(x_train.shape[1:]))

ho.add_tensor("x_train", x_train)
ho.add_tensor("y_train", y_train)
ho.add_tensor("x_test", x_test)
ho.add_tensor("y_test", y_test)

bundle_path = "handover_mnist_1_to_2.tar.gz"
ho.pack(bundle_path)

print(f"✅ Step 1 complete. Handover bundle packed: {bundle_path}\n")
