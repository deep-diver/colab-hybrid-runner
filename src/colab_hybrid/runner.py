import os
import inspect
import json
import subprocess
import functools
from typing import Callable, Any, List, Dict
try:
    from .handover import HandoverManager
except ImportError:
    from handover import HandoverManager

# Global registry for cell functions for exporting to notebook
REGISTERED_CELLS: List[Dict[str, Any]] = []

def hybrid_cell(target: str = "local", session_name: str = "hybrid_framework_session", gpu: str = None):
    """
    Transparent decorator that executes a standard Python function either:
    - Locally, OR
    - Remotely on Colab (with automatic argument packing, upload, remote execution, and download)
    """
    def decorator(func: Callable):
        func_name = func.__name__
        source_code = inspect.getsource(func)

        # Register for notebook export
        REGISTERED_CELLS.append({
            "name": func_name,
            "target": target,
            "gpu": gpu,
            "func": func,
            "source_code": source_code
        })

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if target == "local":
                print(f"🔹 [HybridFramework] Executing '{func_name}' LOCALLY...")
                return func(*args, **kwargs)

            elif target == "remote":
                print(f"🚀 [HybridFramework] Executing '{func_name}' REMOTELY on Colab ({gpu or 'CPU'})...")
                
                # 1. Transparently Pack Arguments into Handover Bundle
                ho_in = HandoverManager(f"in_{func_name}")
                for idx, arg in enumerate(args):
                    if hasattr(arg, "shape") and hasattr(arg, "dtype"): # PyTorch Tensor or NumPy
                        import torch
                        if isinstance(arg, torch.Tensor):
                            ho_in.add_tensor(f"arg_{idx}", arg)
                    elif hasattr(arg, "to_parquet"): # Pandas DataFrame
                        ho_in.add_dataframe(f"arg_{idx}", arg)
                    else:
                        ho_in.add_meta(f"arg_{idx}", arg)

                for k, v in kwargs.items():
                    ho_in.add_meta(k, v)

                in_bundle = f"_tmp_{func_name}_in.tar.gz"
                ho_in.pack(in_bundle)

                # 2. Ensure Colab Session is active
                gpu_flag = f"--gpu {gpu}" if gpu else ""
                subprocess.run(f"colab new -s {session_name} {gpu_flag}", shell=True, capture_output=True)

                # 3. Upload Helper Modules & Input Bundle
                subprocess.run(f"colab upload -s {session_name} handover.py /content/handover.py", shell=True, capture_output=True)
                subprocess.run(f"colab upload -s {session_name} {in_bundle} /content/{in_bundle}", shell=True, capture_output=True)

                # 4. Generate Remote Runner Script
                remote_script = f"_tmp_remote_run_{func_name}.py"
                out_bundle = f"_tmp_{func_name}_out.tar.gz"
                
                # Clean source code (strip decorator for remote execution)
                clean_code = "\n".join([line for line in source_code.split("\n") if not line.strip().startswith("@hybrid_cell")])

                remote_runner_code = f"""
import torch
import pandas as pd
from handover import HandoverManager

{clean_code}

# Unpack inputs
ho_in = HandoverManager.unpack("/content/{in_bundle}")
args = []
idx = 0
while f"arg_{{idx}}" in ho_in.tensors or f"arg_{{idx}}" in ho_in.tables or f"arg_{{idx}}" in ho_in.metadata:
    if f"arg_{{idx}}" in ho_in.tensors:
        args.append(ho_in.tensors[f"arg_{{idx}}"])
    elif f"arg_{{idx}}" in ho_in.tables:
        args.append(ho_in.tables[f"arg_{{idx}}"])
    elif f"arg_{{idx}}" in ho_in.metadata:
        args.append(ho_in.metadata[f"arg_{{idx}}"])
    idx += 1

# Execute target function
res = {func_name}(*args)

# Pack output
ho_out = HandoverManager("out_{func_name}")
if isinstance(res, tuple):
    for i, item in enumerate(res):
        if isinstance(item, torch.Tensor):
            ho_out.add_tensor(f"out_{{i}}", item.cpu())
        elif hasattr(item, "state_dict"): # PyTorch Model
            for k, v in item.state_dict().items():
                ho_out.add_tensor(f"model_{{k}}", v.cpu())
        elif hasattr(item, "to_parquet"):
            ho_out.add_dataframe(f"out_{{i}}", item)
        else:
            ho_out.add_meta(f"out_{{i}}", item)
elif isinstance(res, torch.Tensor):
    ho_out.add_tensor("out_0", res.cpu())
elif hasattr(res, "state_dict"):
    for k, v in res.state_dict().items():
        ho_out.add_tensor(f"model_{{k}}", v.cpu())
else:
    ho_out.add_meta("out_0", res)

ho_out.pack("/content/{out_bundle}")
"""
                with open(remote_script, "w") as f:
                    f.write(remote_runner_code)

                # Execute Remote Runner Script
                subprocess.run(f"colab exec -s {session_name} -f {remote_script}", shell=True)

                # 5. Download Output Bundle
                subprocess.run(f"colab download -s {session_name} /content/{out_bundle} ./{out_bundle}", shell=True, capture_output=True)

                # 6. Unpack & Return Results to Local Context
                ho_out = HandoverManager.unpack(out_bundle)
                
                # Cleanup temp files
                for f_tmp in [in_bundle, remote_script, out_bundle]:
                    if os.path.exists(f_tmp):
                        os.remove(f_tmp)

                if "out_0" in ho_out.tensors:
                    return ho_out.tensors["out_0"]
                elif "out_0" in ho_out.metadata:
                    return ho_out.metadata["out_0"]
                elif any(k.startswith("model_") for k in ho_out.tensors):
                    # Model state dict
                    return {k[6:]: v for k, v in ho_out.tensors.items() if k.startswith("model_")}
                return ho_out.metadata

        return wrapper
    return decorator


def export_to_notebook(output_ipynb_path: str = "exported_notebook.ipynb"):
    """
    Exports all registered cell functions into a clean, standalone Jupyter Notebook (.ipynb)
    with ZERO handover code clutter!
    """
    cells = []
    
    # Title Markdown Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Auto-Generated Clean Pipeline Notebook\n",
            "This notebook contains pure cell logic without any hybrid/handover code."
        ]
    })

    for item in REGISTERED_CELLS:
        # Strip @hybrid_cell decorator
        clean_lines = [line + "\n" for line in item["source_code"].split("\n") if not line.strip().startswith("@hybrid_cell")]
        
        # Markdown Header
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"### Step: `{item['name']}` (Target: {item['target']})"]
        })
        
        # Code Cell
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": clean_lines
        })

    notebook = {
        "cells": cells,
        "metadata": {
            "language_info": {"name": "python"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open(output_ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)

    print(f"✨ [NotebookExporter] Successfully exported clean notebook to: {output_ipynb_path}")
