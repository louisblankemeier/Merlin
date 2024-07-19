import os

from huggingface_hub import hf_hub_download
import torch
from torch import nn

import merlin


class Merlin(nn.Module):
    def __init__(self):
        super(Merlin, self).__init__()
        self.current_path = os.path.dirname(os.path.abspath(__file__))
        self.local_dir = os.path.join(self.current_path, "checkpoints")
        self.checkpoint_name = (
            "i3_resnet_clinical_longformer_best_clip_04-02-2024_23-21-36_epoch_99.pt"
        )
        self.repo_id = "louisblankemeier/Merlin"
        self.model = self._load_model()

    def _load_model(self):
        self._download_checkpoint()
        model = merlin.models.build.MerlinArchitecture()
        model.load_state_dict(
            torch.load(os.path.join(self.local_dir, self.checkpoint_name))
        )
        return model

    def _download_checkpoint(self):
        merlin.utils.download_file(
            repo_id=self.repo_id,
            filename=self.checkpoint_name,
            local_dir=self.local_dir,
        )

    def forward(self, *input):
        return self.model(*input)
