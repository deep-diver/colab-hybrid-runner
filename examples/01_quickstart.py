import torch
from hybrid_runner import hybrid_cell, export_to_notebook

@hybrid_cell(target="local")
def step1_prepare_data():
    print("Preparing vector data locally...")
    x = torch.tensor([10.0, 20.0, 30.0, 40.0])
    return x

@hybrid_cell(target="remote", gpu="T4")
def step2_gpu_compute(x):
    # Pure PyTorch logic on GPU (Zero handover clutter!)
    print(f"Colab GPU received tensor: {x}")
    result = x * 2.5 + 100.0
    return result

@hybrid_cell(target="local")
def step3_local_result(res):
    print("Received computation result on local CPU:")
    print(res)
    return res

if __name__ == "__main__":
    print("==================================================")
    print("🧪 Running Transparent Hybrid Application Demo")
    print("==================================================")

    # 1. Execute workflow transparently
    x_data = step1_prepare_data()
    gpu_res = step2_gpu_compute(x_data)
    step3_local_result(gpu_res)

    # 2. Export clean Jupyter Notebook without handover code
    export_to_notebook("clean_demo_pipeline.ipynb")
    print("==================================================")
