import json
from huggingface_hub import hf_hub_download, HfApi

repo_id = "eternalmay33/pick_place_test"

# 1. Download the info.json from the Hub to see what version is actually listed
print(f"Checking version inside {repo_id}...")
file_path = hf_hub_download(repo_id=repo_id, filename="meta/info.json", repo_type="dataset")

with open(file_path, 'r') as f:
    data = json.load(f)
    true_version = data.get("codebase_version")

print(f"\nFOUND VERSION: {true_version}")

# 2. Apply the correct tag if it's missing
if true_version:
    api = HfApi()
    api.create_tag(repo_id, tag=true_version, repo_type="dataset")
    print(f"SUCCESS: Tagged {repo_id} with {true_version}")
else:
    print("ERROR: Could not find 'codebase_version' in info.json")