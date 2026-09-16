"""
setup_ollama.py
Installs Ollama, starts its server in the background, and pulls
the Qwen 2.5 3B model. Run once per Colab session:
    !python setup_ollama.py
"""

import subprocess
import time
import requests

MODEL_NAME = "qwen2.5:3b"


def install_ollama():
    print("Installing Ollama...")
    subprocess.run(
        "curl -fsSL https://ollama.com/install.sh | sh",
        shell=True,
        check=True,
    )


def start_server():
    print("Starting Ollama server in background...")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for i in range(60):
        try:
            requests.get("http://localhost:11434", timeout=2)
            print("Ollama server is up.")
            return
        except Exception:
            time.sleep(2)
    raise RuntimeError("Ollama server did not start in time after 120s.")


def pull_model():
    print(f"Pulling {MODEL_NAME} (this may take a few minutes)...")
    subprocess.run(["ollama", "pull", MODEL_NAME], check=True)
    print("Model ready.")


if __name__ == "__main__":
    install_ollama()
    start_server()
    pull_model()
