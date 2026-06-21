import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'MedCLIP-main', 'MedCLIP-main'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from medclip.modeling_medclip import MedCLIPVisionModelViT

MEDCLIP_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class Adapter(nn.Module):

    def __init__(self, in_dim: int = 512, bottleneck_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.pre_norm  = nn.LayerNorm(in_dim)
        self.down      = nn.Linear(in_dim, bottleneck_dim, bias=False)
        self.bn        = nn.BatchNorm1d(bottleneck_dim)
        self.act       = nn.GELU()
        self.drop      = nn.Dropout(dropout)
        self.up        = nn.Linear(bottleneck_dim, in_dim, bias=False)
        self.post_norm = nn.LayerNorm(in_dim)
        # learnable scale — initialised near zero so adapter starts as identity
        self.scale     = nn.Parameter(torch.zeros(1))

        nn.init.kaiming_normal_(self.down.weight, nonlinearity='relu')
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        residual = x
        h = self.pre_norm(x)
        h = self.down(h)
        h = self.bn(h)
        h = self.act(h)
        h = self.drop(h)
        h = self.up(h)
        h = F.normalize(h, dim=-1)          # L2-normalise before scaling
        h = h * self.scale.tanh()            # bounded learnable scale
        return self.post_norm(residual + h)


class AgeHead(nn.Module):
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


class SexHead(nn.Module):
    def __init__(self, in_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128),   nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


class AgeGenderModel(nn.Module):
    def __init__(self, medclip_checkpoint: str | None = None):
        super().__init__()
        self.encoder = MedCLIPVisionModelViT(medclip_checkpoint=medclip_checkpoint)
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.adapter  = Adapter(in_dim=512, bottleneck_dim=128)
        self.age_head = AgeHead(in_dim=512)
        self.sex_head = SexHead(in_dim=512)

    def forward(self, pixel_values):
        feat     = self.encoder(pixel_values, project=True)  # (B, 512)
        feat     = self.adapter(feat)
        age_pred = self.age_head(feat).squeeze(-1)            # (B,)
        sex_pred = self.sex_head(feat)                        # (B, 2)
        return age_pred, sex_pred
