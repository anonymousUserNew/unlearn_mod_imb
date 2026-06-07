import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from torchvision.models import resnet50, ResNet50_Weights


class MultimodalFoodClassifier(nn.Module):
    """
    End-to-end Multimodal Classifier for Food101 (101 classes)
    Image Encoder: ResNet50
    Text Encoder: BERT-base-uncased
    Fusion: Feature concatenation
    """

    def __init__(self, num_classes=50, dropout=0.3):
        super().__init__()

        # -----------------------------
        # TEXT ENCODER (BERT)
        # -----------------------------
        self.text_encoder = BertModel.from_pretrained(
            "bert-base-uncased"
        )

        # -----------------------------
        # IMAGE ENCODER (ResNet50)
        # -----------------------------
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(
            *list(backbone.children())[:-1]  # remove FC layer
        )

        # -----------------------------
        # PROJECTION LAYERS
        # -----------------------------
        self.text_proj = nn.Linear(768, 512)
        self.image_proj = nn.Linear(2048, 512)

        # -----------------------------
        # CLASSIFIER
        # -----------------------------
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 512)
        )

        self.fusion_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.text_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )
        self.image_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(
        self,
        image,
        input_ids,
        attention_mask,
        use_image=True,
        use_text=True,
        return_intermediate=False
    ):
        batch_size = image.size(0)
        device = image.device


        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_emb = text_outputs.last_hidden_state[:, 0]  # CLS token
        text_emb = self.text_proj(text_emb)
        text_emb = F.normalize(text_emb, dim=1)


        image_emb = self.image_encoder(image)
        image_emb = image_emb.flatten(start_dim=1)
        image_emb = self.image_proj(image_emb)
        image_emb = F.normalize(image_emb, dim=1)

        # -------- FUSION LOGITS--------
        fused_emb = torch.cat([text_emb, image_emb], dim=1)
        fused_emb = self.fusion(fused_emb)
        fused_emb = F.normalize(fused_emb, dim=1)
        fusion_logits = self.fusion_classifier(fused_emb)

        # -------- TEXT LOGITS --------
        text_logits = self.text_classifier(text_emb)


        # -------- IMAGE LOGITS --------
        image_logits = self.image_classifier(image_emb)

        if return_intermediate:
            return {
                "fusion_logits": fusion_logits,
                "text_logits": text_logits,
                "image_logits": image_logits,
                "text_emb": text_emb,
                "image_emb": image_emb,
                "fused_emb": fused_emb
            }
        return fusion_logits
