"""
Base Utilities for Food101 Unlearning Experiments
================================================
Common helper functions used across all unlearning methods.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# Discover project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model_new_r import MultimodalFoodClassifier


def load_model(checkpoint_path, num_classes, device):
    """
    Load the MultimodalFoodClassifier from a checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint
        num_classes: Number of classes (101 for Food101)
        device: torch.device

    Returns:
        model: MultimodalFoodClassifier instance loaded to device
    """
    model = MultimodalFoodClassifier(num_classes=num_classes)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


@torch.no_grad()
def evaluate(model, loader, device, split_name="Test"):
    """
    Evaluate model on a dataset and return accuracy matrix.

    Returns:
        matrix: np.ndarray of shape (2, 3)
            Row 0: Forget class accuracy [text, image, fusion]
            Row 1: Retain class accuracy [text, image, fusion]
    """
    model.eval()

    # Track accuracy per modality
    text_correct = 0
    image_correct = 0
    fusion_correct = 0
    total = 0

    for batch in tqdm(loader, desc=f"  Evaluating {split_name}"):
        images = batch['image'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        out = model(images, input_ids, attention_mask, return_intermediate=True)

        # Get predictions from each branch
        text_preds   = out['text_logits'].argmax(dim=1)
        image_preds  = out['image_logits'].argmax(dim=1)
        fusion_preds = out['fusion_logits'].argmax(dim=1)

        # Accumulate correct predictions
        text_correct   += (text_preds == labels).sum().item()
        image_correct  += (image_preds == labels).sum().item()
        fusion_correct += (fusion_preds == labels).sum().item()
        total          += labels.size(0)

    # Calculate accuracies
    text_acc   = text_correct  / total * 100
    image_acc  = image_correct / total * 100
    fusion_acc = fusion_correct / total * 100

    print(f"\n  {split_name} Accuracy:")
    print(f"    Text  : {text_acc:.2f}%")
    print(f"    Image : {image_acc:.2f}%")
    print(f"    Fusion: {fusion_acc:.2f}%")

    # Return as matrix format for compatibility
    matrix = np.array([
        [text_acc, image_acc, fusion_acc],  # Row 0
        [text_acc, image_acc, fusion_acc],  # Row 1
    ])

    return matrix


@torch.no_grad()
def evaluate_split(model, forget_loader, retain_loader, device):
    """
    Evaluate model separately on forget and retain sets.

    Returns:
        matrix: np.ndarray of shape (2, 3)
            Row 0: Forget class accuracy [text, image, fusion]
            Row 1: Retain class accuracy [text, image, fusion]
    """
    model.eval()

    def get_accs(loader, name):
        text_correct   = 0
        image_correct  = 0
        fusion_correct = 0
        total          = 0

        for batch in tqdm(loader, desc=f"  Evaluating {name}"):
            images = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            out = model(images, input_ids, attention_mask, return_intermediate=True)

            text_preds   = out['text_logits'].argmax(dim=1)
            image_preds  = out['image_logits'].argmax(dim=1)
            fusion_preds = out['fusion_logits'].argmax(dim=1)

            text_correct   += (text_preds == labels).sum().item()
            image_correct  += (image_preds == labels).sum().item()
            fusion_correct += (fusion_preds == labels).sum().item()
            total          += labels.size(0)

        if total == 0:
            return np.array([0.0, 0.0, 0.0])

        return np.array([
            text_correct   / total * 100,
            image_correct  / total * 100,
            fusion_correct / total * 100,
        ])

    forget_accs = get_accs(forget_loader, "Forget")
    retain_accs = get_accs(retain_loader, "Retain")

    matrix = np.stack([forget_accs, retain_accs], axis=0)

    return matrix


def print_accuracy_matrix(matrix, title="Accuracy Matrix"):
    """
    Pretty print the accuracy matrix.

    Args:
        matrix: shape (2, 3)
            Row 0: Forget [text, image, fusion]
            Row 1: Retain [text, image, fusion]
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    print(f"{'':15} {'Text':>12} {'Image':>12} {'Fusion':>12}")
    print(f"{'-'*70}")
    print(f"{'Forget':15} {matrix[0,0]:11.2f}% {matrix[0,1]:11.2f}% {matrix[0,2]:11.2f}%")
    print(f"{'Retain':15} {matrix[1,0]:11.2f}% {matrix[1,1]:11.2f}% {matrix[1,2]:11.2f}%")
    print(f"{'='*70}\n")


def save_checkpoint(model, save_path, epoch=0, val_acc=0.0):
    """
    Save model checkpoint.

    Args:
        model: MultimodalFoodClassifier instance
        save_path: Path to save checkpoint
        epoch: Current epoch number
        val_acc: Validation accuracy
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    torch.save({
        'epoch': epoch,
        'val_acc': val_acc,
        'model_state_dict': model.state_dict(),
    }, save_path)

    print(f"Checkpoint saved -> {save_path}")