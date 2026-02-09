import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import torch
from tqdm import tqdm
