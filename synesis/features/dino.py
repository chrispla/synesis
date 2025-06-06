"""DINO.
https://github.com/facebookresearch/dino
License: Apache-2.0 (see NOTICE for full license)
"""

import torch
from torch import nn
from transformers import ResNetModel


class DINO(nn.Module):
    def __init__(self, feature_extractor=True):
        super(DINO, self).__init__()

        self.feature_extractor = feature_extractor

        self.model = ResNetModel.from_pretrained("Ramos-Ramos/dino-resnet-50")

    def forward(self, x):
        if self.feature_extractor:
            with torch.no_grad():
                h = self.model(x).last_hidden_state
                h = h.mean(dim=[-1, -2])
                return h
        else:
            raise NotImplementedError("Training not implemented yet.")
