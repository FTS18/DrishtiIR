"""
mock_weights.py
---------------
Generates a mock `generator_latest.pth` checkpoint using randomly initialized
Generator weights. This allows the Streamlit dashboard (app.py) to launch and
run in full demonstration mode without requiring a full training run.

Usage:
    python mock_weights.py
"""

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.model import Generator

def generate_mock_checkpoint(output_dir: str = "checkpoints") -> None:
    os.makedirs(output_dir, exist_ok=True)
    gen = Generator()
    path = os.path.join(output_dir, "generator_latest.pth")
    torch.save(gen.state_dict(), path)
    print(f"Mock checkpoint saved -> {path}")
    print("You can now launch the dashboard with:  streamlit run app.py")

if __name__ == "__main__":
    generate_mock_checkpoint()
