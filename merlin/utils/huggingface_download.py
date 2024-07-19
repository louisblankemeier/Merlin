import os

from huggingface_hub import hf_hub_download


def download_file(
    repo_id: str,
    filename: str,
    local_dir: str,
):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = hf_hub_download(
        repo_id=repo_id, filename=filename, local_dir=local_dir
    )
    print(f"{filename} downloaded and saved to {local_file_path}")
    return local_file_path
