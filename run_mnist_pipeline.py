import os
import sys
import subprocess

SESSION_NAME = "mnist_hybrid_session"

def run_cmd(cmd_str: str) -> str:
    print(f"--> Command: {cmd_str}")
    res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"⚠️ Command Error (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
    return res.stdout.strip()

def main():
    print("=========================================================")
    print("🚀 Starting MNIST Hybrid Pipeline Orchestration")
    print("   1. Data Preprocessing (Local)")
    print("   2. Model Training (Colab T4 GPU)")
    print("   3. Model Evaluation (Colab T4 GPU)")
    print("   4. Inference Test (Local CPU)")
    print("=========================================================\n")

    # Step 1: Local Preprocessing
    print("Step 1/4: Running Local Preprocessing...")
    out1 = run_cmd("PYTHONPATH=. python3 cells/01_mnist_prep.py")
    print(out1)

    # Step 2: Colab Provisioning & Uploads
    print("\nStep 2/4: Provisioning Colab Session & Uploading Assets...")
    run_cmd(f"colab new -s {SESSION_NAME} --gpu T4")
    run_cmd(f"colab upload -s {SESSION_NAME} handover.py /content/handover.py")
    run_cmd(f"colab upload -s {SESSION_NAME} mnist_model.py /content/mnist_model.py")
    run_cmd(f"colab upload -s {SESSION_NAME} cells/02_mnist_train.py /content/cells/02_mnist_train.py")
    run_cmd(f"colab upload -s {SESSION_NAME} cells/03_mnist_eval.py /content/cells/03_mnist_eval.py")
    run_cmd(f"colab upload -s {SESSION_NAME} handover_mnist_1_to_2.tar.gz /content/handover_mnist_1_to_2.tar.gz")

    # Step 3: Remote Training on Colab GPU
    print("\nStep 2/4: Running Remote Training on Colab GPU...")
    out2 = run_cmd(f"colab exec -s {SESSION_NAME} -f cells/02_mnist_train.py")
    print(out2)

    # Step 4: Remote Evaluation on Colab GPU
    print("\nStep 3/4: Running Remote Evaluation on Colab GPU...")
    out3 = run_cmd(f"colab exec -s {SESSION_NAME} -f cells/03_mnist_eval.py")
    print(out3)

    # Step 5: Download & Stop Session
    print("\nStep 4/4: Downloading Model Checkpoint & Stopping Colab Session...")
    run_cmd(f"colab download -s {SESSION_NAME} /content/handover_mnist_3_to_4.tar.gz ./handover_mnist_3_to_4.tar.gz")
    run_cmd(f"colab stop -s {SESSION_NAME}")

    # Step 6: Local Inference Test
    print("\nFinal Step: Running Local Inference Test...")
    out4 = run_cmd("PYTHONPATH=. python3 cells/04_mnist_infer.py")
    print(out4)

if __name__ == "__main__":
    main()
