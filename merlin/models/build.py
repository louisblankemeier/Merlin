import copy

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
import torchvision

from merlin.models import i3res


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet152(pretrained=True)
        self.i3_resnet = i3res.I3ResNet(copy.deepcopy(resnet), class_nb=1692, conv_class=True)

    def forward(self, image):
        contrastive_features, ehr_features = self.i3_resnet(image)
        return contrastive_features, ehr_features


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "yikuan8/Clinical-Longformer"
        )
        self.text_encoder = AutoModel.from_pretrained("yikuan8/Clinical-Longformer")
        self.text_encoder.gradient_checkpointing_enable()
        self.linear_layer = nn.Linear(768, 512)

    def forward(self, text_labels):
        text_labels = [text.lower() for text in text_labels]
        inputs = self.tokenizer(
            text_labels,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.text_encoder.device) for k, v in inputs.items()}
        text_embeddings = self.text_encoder(**inputs).last_hidden_state[:, 0, :]
        text_embeddings = self.linear_layer(text_embeddings)
        return text_embeddings


class MerlinArchitecture(nn.Module):
    def __init__(self, init_logit_scale: float = 1.0):
        super().__init__()
        self.encode_image = ImageEncoder()
        self.encode_text = TextEncoder()
        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)

    def forward(self, image, text):
        image_features, ehr_features = self.encode_image(image)
        text_features = self.encode_text(text)

        if len(image_features.shape) == 1:
            image_features = image_features.unsqueeze(0)
        if len(text_features.shape) == 1:
            text_features = text_features.unsqueeze(0)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return (
            image_features,
            ehr_features,
            text_features,
        )
