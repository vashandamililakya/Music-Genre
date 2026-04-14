"""
train_model.py
Standalone training script in project root — used by Render build.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from model.train import train

if __name__ == "__main__":
    train(demo=True)
    print("Model trained and saved successfully.")
