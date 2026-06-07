"""
DCASE Multimodal Model
=======================
Architecture:
  - Video Encoder : ResNet50 (pre-trained ImageNet) → 2048-d features
  - Audio Encoder : ResNet50 (pre-trained ImageNet) → 2048-d features
                     (audio is fed as 3-ch mel-spectrogram image)
  - 3 Branches:
      · video_branch  : video_emb → classifier  → video_logits
      · audio_branch  : audio_emb → classifier  → audio_logits
      · fusion_branch : cat(video_emb, audio_emb) → fusion → classifier → fusion_logits

10 scene classes for DCASE acoustic scene classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNetEncoder(nn.Module):
    """
    Shared building block: a ResNet50 backbone stripped of its final FC layer.
    Output: (B, 2048) feature vector.
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights        = ResNet50_Weights.DEFAULT if pretrained else None
        backbone       = resnet50(weights=weights)
        self.encoder   = nn.Sequential(*list(backbone.children())[:-1])  # → (B, 2048, 1, 1)
        self.out_dim   = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224) → (B, 2048)"""
        feat = self.encoder(x)           # (B, 2048, 1, 1)
        return feat.flatten(start_dim=1) # (B, 2048)


class DcaseMultimodalModel(nn.Module):
    """
    Dual-ResNet50 model for DCASE audio-video scene classification.

    :param num_classes: Number of target classes (10 for DCASE).
    :param embed_dim:   Projected embedding size (default 512).
    :param dropout:     Dropout probability in classifiers.
    """

    def __init__(self, num_classes: int = 10, embed_dim: int = 512, dropout: float = 0.3):
        super().__init__()

        # ── Encoders ──────────────────────────────────────────────────────────
        self.video_encoder = ResNetEncoder(pretrained=True)
        self.audio_encoder = ResNetEncoder(pretrained=True)

        # ── Projection (2048 → embed_dim) ─────────────────────────────────────
        self.video_proj = nn.Linear(2048, embed_dim)
        self.audio_proj = nn.Linear(2048, embed_dim)

        # ── Fusion MLP ────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

        # ── Three Classification Heads ────────────────────────────────────────
        def _head():
            return nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes),
            )

        self.fusion_classifier = _head()
        self.video_classifier  = _head()
        self.audio_classifier  = _head()

    # ────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        video: torch.Tensor,
        spectrogram: torch.Tensor,
        return_intermediate: bool = False,
    ):
        """
        Args:
            video        : (B, 3, 224, 224) – video frame
            spectrogram  : (B, 3, 224, 224) – mel-spectrogram of audio
            return_intermediate : if True, return dict with all logits + embeddings

        Returns:
            fusion_logits (B, C)  or  dict with full intermediates.
        """
        # ── Encode ────────────────────────────────────────────────────────────
        vid_raw = self.video_encoder(video)          # (B, 2048)
        aud_raw = self.audio_encoder(spectrogram)    # (B, 2048)

        # ── Project & normalise ───────────────────────────────────────────────
        vid_emb = F.normalize(self.video_proj(vid_raw), dim=1)   # (B, 512)
        aud_emb = F.normalize(self.audio_proj(aud_raw), dim=1)   # (B, 512)

        # ── Fusion ────────────────────────────────────────────────────────────
        fused_emb = torch.cat([vid_emb, aud_emb], dim=1)         # (B, 1024)
        fused_emb = self.fusion(fused_emb)                        # (B, 512)
        fused_emb = F.normalize(fused_emb, dim=1)

        # ── Logits ────────────────────────────────────────────────────────────
        fusion_logits = self.fusion_classifier(fused_emb)   # (B, C)
        video_logits  = self.video_classifier(vid_emb)      # (B, C)
        audio_logits  = self.audio_classifier(aud_emb)      # (B, C)

        if return_intermediate:
            return {
                "fusion_logits": fusion_logits,
                "video_logits" : video_logits,
                "audio_logits" : audio_logits,
                "vid_emb"      : vid_emb,
                "aud_emb"      : aud_emb,
                "fused_emb"    : fused_emb,
            }
        return fusion_logits


# Backwards-compat alias (in case any script still uses the old name)
RavdessMultimodalModel = DcaseMultimodalModel
