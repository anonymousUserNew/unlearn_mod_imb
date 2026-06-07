"""
Base Utilities for ADVANCE Unlearning Experiments
==================================================
Common helper functions used across all unlearning methods.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import AdvanceMultimodalModel


def load_model(checkpoint_path, num_classes, device):
    """
    Load the AdvanceMultimodalModel from a checkpoint.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        num_classes: Number of classes (13 for ADVANCE)
        device: torch.device
        
    Returns:
        model: AdvanceMultimodalModel instance loaded to device
    """
    model = AdvanceMultimodalModel(num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
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
            Row 0: Forget class accuracy [audio, image, fusion]
            Row 1: Retain class accuracy [audio, image, fusion]
    
    Note: For this to work properly, the loader should contain both
    forget and retain samples with proper class labels.
    """
    model.eval()
    
    # Track accuracy per modality
    audio_correct = 0
    image_correct = 0
    fusion_correct = 0
    total = 0
    
    # Track forget vs retain separately
    # We'll need to know which samples are forget class
    # For now, we'll just compute overall accuracy
    
    for batch in tqdm(loader, desc=f"  Evaluating {split_name}"):
        images = batch['image'].to(device)
        specs = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        out = model(images, specs, return_intermediate=True)
        
        # Get predictions from each branch
        audio_preds = out['audio_logits'].argmax(dim=1)
        image_preds = out['image_logits'].argmax(dim=1)
        fusion_preds = out['fusion_logits'].argmax(dim=1)
        
        # Accumulate correct predictions
        audio_correct += (audio_preds == labels).sum().item()
        image_correct += (image_preds == labels).sum().item()
        fusion_correct += (fusion_preds == labels).sum().item()
        total += labels.size(0)
    
    # Calculate accuracies
    audio_acc = audio_correct / total * 100
    image_acc = image_correct / total * 100
    fusion_acc = fusion_correct / total * 100
    
    print(f"\n  {split_name} Accuracy:")
    print(f"    Audio : {audio_acc:.2f}%")
    print(f"    Image : {image_acc:.2f}%")
    print(f"    Fusion: {fusion_acc:.2f}%")
    
    # Return as matrix format for compatibility
    # For now, return same values for both rows (will be improved in specific methods)
    matrix = np.array([
        [audio_acc, image_acc, fusion_acc],  # Row 0
        [audio_acc, image_acc, fusion_acc],  # Row 1
    ])
    
    return matrix


@torch.no_grad()
def evaluate_split(model, forget_loader, retain_loader, device):
    """
    Evaluate model separately on forget and retain sets.
    
    Returns:
        matrix: np.ndarray of shape (2, 3)
            Row 0: Forget class accuracy [audio, image, fusion]
            Row 1: Retain class accuracy [audio, image, fusion]
    """
    model.eval()
    
    def get_accs(loader, name):
        audio_correct = 0
        image_correct = 0
        fusion_correct = 0
        total = 0
        
        for batch in tqdm(loader, desc=f"  Evaluating {name}"):
            images = batch['image'].to(device)
            specs = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            out = model(images, specs, return_intermediate=True)
            
            audio_preds = out['audio_logits'].argmax(dim=1)
            image_preds = out['image_logits'].argmax(dim=1)
            fusion_preds = out['fusion_logits'].argmax(dim=1)
            
            audio_correct += (audio_preds == labels).sum().item()
            image_correct += (image_preds == labels).sum().item()
            fusion_correct += (fusion_preds == labels).sum().item()
            total += labels.size(0)
        
        if total == 0:
            return np.array([0.0, 0.0, 0.0])
        
        return np.array([
            audio_correct / total * 100,
            image_correct / total * 100,
            fusion_correct / total * 100
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
            Row 0: Forget [audio, image, fusion]
            Row 1: Retain [audio, image, fusion]
    """
    print(f"\n{'='*70}")
    print(f"{title:^70}")
    print(f"{'='*70}")
    print(f"{'':15} {'Audio':>12} {'Image':>12} {'Fusion':>12}")
    print(f"{'-'*70}")
    print(f"{'Forget':15} {matrix[0,0]:11.2f}% {matrix[0,1]:11.2f}% {matrix[0,2]:11.2f}%")
    print(f"{'Retain':15} {matrix[1,0]:11.2f}% {matrix[1,1]:11.2f}% {matrix[1,2]:11.2f}%")
    print(f"{'='*70}\n")


def save_checkpoint(model, save_path, epoch=0, val_acc=0.0):
    """
    Save model checkpoint.
    
    Args:
        model: AdvanceMultimodalModel instance
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