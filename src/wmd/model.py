from __future__ import annotations
import torch
from torch import nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        num_groups = min(8, out_channels)
        self.block = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False), nn.GroupNorm(num_groups, out_channels), nn.ReLU(inplace=True), nn.MaxPool3d(kernel_size=2))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
class WMDClassifier3D(nn.Module):
    def __init__(self, num_classes: int=2, in_channels: int=1) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(ConvBlock(in_channels, 8), ConvBlock(8, 16), ConvBlock(16, 32), ConvBlock(32, 64))
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(64, 32), nn.ReLU(inplace=True), nn.Linear(32, num_classes))
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
IMAGE_EMBED_DIM = 64
class MultimodalWMDClassifier(nn.Module):
    def __init__(self, num_clinical_features: int, num_classes: int=2, in_channels: int=1, clinical_embed_dim: int=16) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_clinical_features = num_clinical_features
        self.features = nn.Sequential(ConvBlock(in_channels, 8), ConvBlock(8, 16), ConvBlock(16, 32), ConvBlock(32, 64))
        self.pool = nn.AdaptiveMaxPool3d(1)
        self.clinical_encoder = nn.Sequential(nn.Linear(num_clinical_features, clinical_embed_dim), nn.ReLU(inplace=True), nn.Linear(clinical_embed_dim, clinical_embed_dim), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(IMAGE_EMBED_DIM + clinical_embed_dim, 32), nn.ReLU(inplace=True), nn.Linear(32, num_classes))
    def image_embedding(self, volume: torch.Tensor) -> torch.Tensor:
        return torch.flatten(self.pool(self.features(volume)), 1)
    def forward(self, volume: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        img = self.image_embedding(volume)
        clin = self.clinical_encoder(clinical)
        fused = torch.cat([img, clin], dim=1)
        return self.head(fused)
def build_model(num_classes: int=2) -> WMDClassifier3D:
    return WMDClassifier3D(num_classes=num_classes)
def build_multimodal_model(num_clinical_features: int, num_classes: int=2) -> MultimodalWMDClassifier:
    return MultimodalWMDClassifier(num_clinical_features=num_clinical_features, num_classes=num_classes)
