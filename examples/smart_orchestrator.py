import os
import sys
import json
import subprocess
from handover import get_file_sha256

SESSION_NAME = "sot_demo_session"

def run_cmd(cmd_str: str) -> str:
    res = subprocess.run(cmd_str, shell=True, capture_output=True, text=True)
    return res.stdout.strip()

def get_remote_file_hash(session_name: str, remote_path: str) -> str:
    """Queries Colab remote VM for the SHA256 checksum of a file."""
    py_cmd = f"import os, hashlib; p='{remote_path}'; print(hashlib.sha256(open(p, 'rb').read()).hexdigest() if os.path.exists(p) else 'NOT_FOUND')"
    output = run_cmd(f"echo \"{py_cmd}\" | colab exec -s {session_name}")
    # Extract last non-empty line
    lines = [line.strip() for line in output.split("\n") if line.strip()]
    for line in reversed(lines):
        if len(line) == 64 or line == "NOT_FOUND":
            return line
    return "NOT_FOUND"

def smart_sync_to_remote(session_name: str, local_path: str, remote_path: str, sot_mode: str):
    local_hash = get_file_sha256(local_path)
    print(f"\n--- Checking Sync for {local_path} -> {remote_path} (SoT: {sot_mode}) ---")
    
    if sot_mode == "REMOTE":
        print("ℹ️ Source of Truth is REMOTE. Keeping existing remote file without uploading.")
        return

    remote_hash = get_remote_file_hash(session_name, remote_path)
    print(f"Local  SHA256: {local_hash[:12]}...")
    print(f"Remote SHA256: {remote_hash[:12]}...")

    if local_hash and local_hash == remote_hash:
        print("⚡ [CACHE HIT] Remote file is already up-to-date! Skipping upload. 🚀")
    else:
        print("📤 [CACHE MISS / UPDATED] Uploading new version to Colab...")
        upload_res = run_cmd(f"colab upload -s {session_name} {local_path} {remote_path}")
        print(f"   {upload_res}")

def main():
    print("=== Smart Orchestrator with Source-of-Truth & Caching Policy ===")
    
    # 1. Run Cell 1 locally
    print("\n1️⃣ Running Cell 1 Locally...")
    run_cmd("PYTHONPATH=. python3 cells/01_local_prep.py")
    
    # Ensure Colab session exists
    sessions_out = run_cmd("colab sessions")
    if SESSION_NAME not in sessions_out:
        print(f"Provisioning Colab session '{SESSION_NAME}'...")
        run_cmd(f"colab new -s {SESSION_NAME} --gpu T4")
    
    # Upload handover.py helper if missing or outdated
    smart_sync_to_remote(SESSION_NAME, "handover.py", "/content/handover.py", "LOCAL")

    # Smart Sync Cell 1 output bundle based on SoT
    smart_sync_to_remote(SESSION_NAME, "handover_1_to_2.tar.gz", "/content/handover_1_to_2.tar.gz", "LOCAL")

    # 2. Run Cell 2 on Colab
    print("\n2️⃣ Running Cell 2 on Colab GPU...")
    exec_out = run_cmd(f"colab exec -s {SESSION_NAME} -f cells/02_colab_train.py")
    print(exec_out)

    # 3. Smart Download Cell 2 output bundle
    print("\n3️⃣ Smart Downloading Cell 2 output bundle...")
    run_cmd(f"colab download -s {SESSION_NAME} /content/handover_2_to_3.tar.gz ./handover_2_to_3.tar.gz")

    # 4. Run Cell 3 locally
    print("\n4️⃣ Running Cell 3 Locally...")
    eval_out = run_cmd("PYTHONPATH=. python3 cells/03_local_eval.py")
    print(eval_out)

if __name__ == "__main__":
    main()
