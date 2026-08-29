import os
import sys
import subprocess

SESSION_NAME = "lora_hybrid_tuning_session"

def run_cmd(cmd_str: str) -> str:
    print(f"--> Executing: {cmd_str}")
    res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"⚠️ Command Warning (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
    return res.stdout.strip()

def main():
    print("=========================================================================")
    print("🚀 PRACTICAL HYBRID LoRA FINE-TUNING PIPELINE")
    print("   1. Local CPU Dataset Preprocessing & Compression (90% Reduction)")
    print("   2. Colab T4 GPU Single-Burst LoRA Fine-Tuning")
    print("   3. LoRA Adapter Download (~1.5MB) & Immediate Colab GPU Teardown")
    print("   4. Local Weight Backup & Local CPU Inference Verification")
    print("=========================================================================\n")

    # Step 1: Local CPU Preprocessing
    print("Step 1/4: Running Local Dataset Preprocessing & Compression...")
    out1 = run_cmd("PYTHONPATH=. python3 cells/01_lora_prep_local.py")
    print(out1)

    # Step 2: Colab GPU Provisioning & Upload Assets
    print("\nStep 2/4: Provisioning Colab T4 GPU Session & Uploading Compact Bundle...")
    run_cmd(f"colab new -s {SESSION_NAME} --gpu T4")
    run_cmd(f"colab upload -s {SESSION_NAME} handover.py /content/handover.py")
    run_cmd(f"colab upload -s {SESSION_NAME} lora_gemma_model.py /content/lora_gemma_model.py")
    run_cmd(f"colab upload -s {SESSION_NAME} cells/02_lora_train_remote.py /content/cells/02_lora_train_remote.py")
    run_cmd(f"colab upload -s {SESSION_NAME} handover_lora_1_to_2.tar.gz /content/handover_lora_1_to_2.tar.gz")

    # Step 3: Single-Burst Remote LoRA Fine-Tuning on Colab GPU
    print("\nStep 3/4: Executing Single-Burst LoRA Fine-Tuning on Colab GPU...")
    out2 = run_cmd(f"colab exec -s {SESSION_NAME} -f cells/02_lora_train_remote.py")
    print(out2)

    # Step 4: Download LoRA Adapter & Teardown Colab Session
    print("\nStep 4/4: Downloading LoRA Adapter Weights & Releasing Colab GPU...")
    run_cmd(f"colab download -s {SESSION_NAME} /content/handover_lora_adapter.tar.gz ./handover_lora_adapter.tar.gz")
    run_cmd(f"colab stop -s {SESSION_NAME}")
    print("🔒 Colab GPU VM Terminated. Zero idle costs accrued!")

    # Step 5: Local Backup Verification & CPU Inference
    print("\nFinal Step: Verifying LoRA Adapter Backup & Local CPU Inference...")
    out3 = run_cmd("PYTHONPATH=. python3 cells/03_lora_eval_local.py")
    print(out3)

if __name__ == "__main__":
    main()
