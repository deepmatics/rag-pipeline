#download open rag dataset
from huggingface_hub import snapshot_download

# This explicitly defines it as a dataset
path = snapshot_download(
    repo_id="vectara/open_ragbench", 
    repo_type="dataset"
)
print(f"Downloaded to: {path}")