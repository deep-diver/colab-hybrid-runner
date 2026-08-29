# ⚡ Colab Hybrid Runner

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Colab CLI](https://img.shields.io/badge/Google%20Colab-CLI-orange.svg)](https://github.com/googlecolab/google-colab-cli)

**Colab Hybrid Runner** is a lightweight, zero-boilerplate Python framework that orchestrates hybrid Machine Learning pipelines across your **Local CPU** and **Google Colab Cloud GPU/TPU** runtimes.

Preprocess data locally for free, execute heavy matrix compute on cloud accelerators in short bursts, cache results with SHA256 fingerprints, and export clean, standalone Jupyter Notebooks (`.ipynb`) with a single click.

<p align="center">
  <img src="assets/colab_hybrid_infographic.png" alt="Colab Hybrid Runner Infographic" width="100%">
</p>

---

## 💡 Why Colab Hybrid Runner?

Running entire end-to-end notebooks on cloud GPUs is often wasteful:
* **High Idle Costs**: GPU compute units are burnt while downloading datasets, parsing text, or debugging data pipelines.
* **Preemption & Session Disconnects**: If a Colab session drops, unsaved checkpoints and intermediate datasets vanish.
* **Messy Transfer Code**: Manually juggling `colab upload`, `colab download`, and serialization scripts clutters your clean ML code.

### The Solution
```text
┌──────────────────────────────────────────────────────────────────────────┐
│                   Your Clean Python / PyTorch Code                       │
│             Zero transfer boilerplate. Just standard functions.          │
└────────────────────────────────────┬─────────────────────────────────────┘
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
[1. Hybrid Execution Mode]                       [2. Standalone Export Mode]
- Local CPU handles I/O & preprocessing.         - Strips decorators and exports a
- Colab GPU executes training in short bursts.     clean, standalone `.ipynb` with
- Smart SHA256 caching skips unchanged uploads.    zero handover clutter.
- Backs up lightweight LoRA weights locally.
```

---

## ⚡ Quickstart

### 1. Installation
Install the official Google Colab CLI:
```bash
pip install google-colab-cli
# Authenticate with Google Cloud / Colab API
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

### 2. Write Your Pipeline
Add `@hybrid_cell` on top of your functions. That's it!

```python
import torch
from hybrid_runner import hybrid_cell, export_to_notebook

# Step 1: Preprocess on Local CPU (Free, no GPU idle costs)
@hybrid_cell(target="local")
def step1_preprocess():
    print("Preprocessing synthetic data locally...")
    x = torch.randn(1000, 64)
    return x

# Step 2: Single-burst GPU Training on Colab (e.g. Tesla T4)
@hybrid_cell(target="remote", gpu="T4")
def step2_train(x):
    # This block executes directly on Colab GPU!
    model = torch.nn.Linear(64, 10).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(5):
        loss = model(x.cuda()).sum()
        loss.backward()
        optimizer.step()
    return model

# Step 3: Result received automatically on Local CPU
@hybrid_cell(target="local")
def step3_evaluate(model):
    print("Model received locally on CPU for evaluation:", model)

if __name__ == "__main__":
    data = step1_preprocess()
    trained_model = step2_train(data)
    step3_evaluate(trained_model)

    # Export a clean, production-ready Jupyter Notebook
    export_to_notebook("pipeline.ipynb")
```

---

## 🔬 Empirical Benchmarks & Hardware Insights

We benchmarked real-world LLM operations on **Google Colab Tesla T4 GPUs** comparing GPU-Only vs. CPU-GPU Hybrid workflows.

### 1. Gemma Model Architecture (Vocab Size = 262,144)
* **Embedding Layer**: Offloading the 262k vocab embedding table to CPU RAM saved **3.67 GB of VRAM** with negligible lookup latency difference (0.04ms vs 0.17ms).
* **Single-Token Sampling (Batch=1)**: CPU L3 Cache sampling was **1.68x ~ 2.80x faster** than GPU CUDA kernel sorting due to avoiding CUDA kernel launch overhead!

### 2. Batch Size Scaling & Crossover Point (Llama-3 Vocab 128k)

| Batch Size | Payload Size | GPU-Only Latency | CPU Hybrid Latency | Winning Mode |
| :---: | :---: | :---: | :---: | :---: |
| **Batch = 1** | **0.51 MB** | 2.42 ms | **1.26 ms** | **⚡ CPU Hybrid (1.92x faster)** |
| **Batch = 4** | **2.05 MB** | 3.12 ms | **2.85 ms** | **⚡ CPU Hybrid (1.09x faster)** |
| **Batch = 16** | **8.19 MB** | **4.85 ms** | 8.92 ms | **🚀 GPU-Only (1.84x faster)** |
| **Batch = 64** | **32.77 MB** | **11.50 ms** | 34.80 ms | **🚀 GPU-Only (3.03x faster)** |
| **Batch = 128** | **65.54 MB** | **20.15 ms** | 71.10 ms | **🚀 GPU-Only (3.53x faster)** |

> **Key Finding**: For interactive single-user inference (Batch 1~4), CPU sampling avoids GPU kernel latency. For large-scale batch serving (Batch 16+), GPU parallel throughput overcomes PCIe transfer overhead.

---

## 🛠️ Case Study: Gemma LoRA Fine-Tuning

In a real end-to-end instruction-tuning test:
1. **Local Preprocessing**: 5.24 MB raw instruction dataset was tokenized and packed locally into a 0.51 MB tensor bundle (**90.2% data reduction**).
2. **Colab GPU Single-Burst Training**: Provisioned a Tesla T4 GPU, fine-tuned LoRA adapters (0.96% trainable params) in **2.76 seconds**.
3. **Lightweight Backup**: Only the **0.79 MB LoRA adapter weights** were downloaded to the local machine, and the Colab VM was terminated immediately.
4. **Local CPU Inference**: Loaded the adapter onto the local CPU base model and verified inference successfully.

**Total GPU billable time**: < 30 seconds. **Zero idle costs.**

---

## 📂 Project Structure

```text
colab-hybrid-runner/
├── handover.py                 # HandoverManager: PyTorch/Parquet/.tar.gz pack & unpack
├── hybrid_runner.py            # @hybrid_cell decorator & export_to_notebook()
├── mnist_model.py              # CNN model architecture definition
├── lora_gemma_model.py         # Gemma Transformer with LoRA adapters
├── demo_transparent_app.py     # Minimal transparent demo
├── run_mnist_pipeline.py       # End-to-end MNIST hybrid pipeline
├── run_hybrid_lora_pipeline.py # End-to-end LoRA fine-tuning pipeline
├── visualize_inference.py      # Inference visualization plotter
├── benchmarks/
│   ├── benchmark_gpu_vs_hybrid.py
│   ├── benchmark_gemma.py
│   └── benchmark_batch_scaling.py
└── cells/                      # Modular cell scripts
```

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or pull request.

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
