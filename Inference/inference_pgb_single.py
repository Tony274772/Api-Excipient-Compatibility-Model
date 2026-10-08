"""
Inference script for single-run PGB models on the 24-pairs held-out test set.
Delegates to experiments.pgb.inference for unified inference execution.
"""

import os
import sys

# Allow running from inside the Inference/ folder or root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.pgb.inference import run_inference

if __name__ == "__main__":
    run_inference()
