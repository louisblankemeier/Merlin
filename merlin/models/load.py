import os

from huggingface_hub import hf_hub_download
import torch
from torch import nn

import merlin

def download_checkpoint(
    repo_id: str,
    filename: str,
    local_dir: str,
):
    os.makedirs(local_dir, exist_ok=True)
    local_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
    print(f"Checkpoint {filename} downloaded and saved to {local_file_path}")

class Merlin(nn.Module):
    def __init__(self):
        super(Merlin, self).__init__()
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        self.repo_id = "louisblankemeier/Merlin"
        self.model = self._load_model()

    def _load_model(self):
        self._download_checkpoint()
        model = merlin.models.build.MerlinArchitecture()
        model.load_state_dict(torch.load(os.path.join(self.local_dir, self.checkpoint_name)))
        return model

    def _download_checkpoint(self):
        download_checkpoint(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir
        )

    def forward(self, *input):
        return self.model(*input)


