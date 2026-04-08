#!/bin/bash
# Create mask_res conda env for FiftyOne exploration + masking analysis
set -e

conda create -n mask_res python=3.11 -y
conda activate mask_res

# Core
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install fiftyone
pip install umap-learn scikit-learn scipy
pip install plotly
pip install transformers pillow
pip install tqdm

echo "Done. Activate with: conda activate mask_res"
