"""
UL (Uncertainty Learning) Unlearning for ADVANCE
=================================================
Push model toward maximum uncertainty on forget samples.
Uses KL divergence to uniform distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

sys.path.insert(0, '/home/team2/Unlearning/ADVANCE')

from src.dataset import get_forget_splits, get_retain_splits
from src.model import AdvanceMultimodalModel
from src.labels import NUM_CLASSES

from unimodal_un.base_utils import load_model, evaluate_split, print_accuracy_matrix, save_checkpoint
from mia_v2 import run_mia


def uncertainty_loss(logits, num_classes):
    """
    Uncertainty loss: KL divergence between model output and uniform distribution.
    Forces model to be maximally uncertain on forget samples.
    """
    # Uniform target distribution
    uniform = torch.full_like(
        torch.softmax(logits, dim=1),
        fill_value=1.0 / num_classes
    ).detach()
    
    log_probs = torch.log_softmax(logits, dim=1)
    
    # KL(uniform || model)
    loss = nn.KLDivLoss(reduction='batchmean')(log_probs, uniform)
    return loss


def train_one_epoch_UL(model, forget_loader, optimizer, device, epoch):
    """
    UL training epoch.
    Only forget set is used. No retain set, no true labels.
    """
    model.train()
    
    total_loss = 0.0
    total_samples = 0
    total_max_prob = 0.0
    
    for batch in tqdm(forget_loader, desc=f"  Epoch {epoch} [UL]"):
        imgs = batch['image'].to(device)
        specs = batch['spectrogram'].to(device)
        # Note: labels not used
        
        optimizer.zero_grad()
        
        out = model(imgs, specs, return_intermediate=True)
        
        # Uncertainty loss on all three branches
        loss = (uncertainty_loss(out['audio_logits'], NUM_CLASSES) +
                uncertainty_loss(out['image_logits'], NUM_CLASSES) +
                uncertainty_loss(out['fusion_logits'], NUM_CLASSES))
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * imgs.size(0)
        total_samples += imgs.size(0)
        
        # Track max prob (should approach 1/num_classes)
        with torch.no_grad():
            max_prob = torch.softmax(out['fusion_logits'], dim=1).max(dim=1).values.mean().item()
            total_max_prob += max_prob * imgs.size(0)
    
    n = total_samples
    avg_max_prob = total_max_prob / n
    perfect_uncertainty = 1.0 / NUM_CLASSES
    
    print(f"\n  Epoch {epoch} Train Summary:")
    print(f"    Uncertainty Loss        : {total_loss/n:.4f}")
    print(f"    Avg Max Prob (fusion)   : {avg_max_prob:.4f}")
    print(f"    Perfect Uncertainty     : {perfect_uncertainty:.4f}")
    print(f"    Gap from perfect        : {abs(avg_max_prob - perfect_uncertainty):.4f}")


def run_UL_unlearning(original_ckpt, save_ckpt_path, forget_class,
                      device, lr=1e-4, epochs=15, batch_size=32,
                      num_workers=4, patience=5):
    """
    Run UL unlearning on ADVANCE dataset.
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    forget_train, forget_val, forget_test = get_forget_splits(
        forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
    )
    retain_train, retain_val, retain_test = get_retain_splits(
        forget_class=forget_class, val_ratio=0.2, test_ratio=0.1, seed=42
    )
    
    print(f"\n  Forget class: {forget_class}")
    print(f"  Forget train: {len(forget_train)}")
    print(f"  Retain test : {len(retain_test)}")
    
    forget_train_loader = DataLoader(forget_train, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    print("\n" + "="*70)
    print("LOADING ORIGINAL MODEL")
    print("="*70)
    
    model = load_model(original_ckpt, NUM_CLASSES, device)
    
    print("\nEvaluating BEFORE unlearning...")
    matrix_before = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_before, title="BEFORE UNLEARNING")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n" + "="*70)
    print("UL UNLEARNING")
    print("="*70)
    
    best_retain_acc = 0.0
    best_epoch = 0
    patience_count = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'-'*70}")
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"{'-'*70}")
        
        train_one_epoch_UL(model, forget_train_loader, optimizer, device, epoch)
        scheduler.step()
        
        matrix_val = evaluate_split(model, forget_test_loader, retain_test_loader, device)
        print_accuracy_matrix(matrix_val, title=f"EPOCH {epoch}")
        
        retain_fusion_acc = matrix_val[1, 2]
        
        if retain_fusion_acc > best_retain_acc:
            best_retain_acc = retain_fusion_acc
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(model, save_ckpt_path, epoch, retain_fusion_acc)
            print(f"✓ New best (epoch {epoch}, retain={retain_fusion_acc:.2f}%)")
        else:
            patience_count += 1
            print(f"  No improvement. Patience {patience_count}/{patience}")
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                break
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETED! Best: epoch {best_epoch}, acc={best_retain_acc:.2f}%")
    print("="*70)
    
    model = load_model(save_ckpt_path, NUM_CLASSES, device)
    matrix_after = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_after, title="AFTER UNLEARNING")
    
    # MIA
    print("\n" + "="*70)
    print("MIA EVALUATION")
    print("="*70)
    
    from src.dataset import RetainDataset, ForgetDataset
    full_retain = RetainDataset(forget_class=forget_class)
    full_forget = ForgetDataset(forget_class=forget_class)
    full_retain_loader = DataLoader(full_retain, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    full_forget_loader = DataLoader(full_forget, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    print("\n[Original Model]")
    orig_model = load_model(original_ckpt, NUM_CLASSES, device)
    run_mia(orig_model, full_retain_loader, retain_test_loader,
            full_forget_loader, device, label="Original")
    
    print("\n[Unlearned Model - UL]")
    run_mia(model, full_retain_loader, retain_test_loader,
            full_forget_loader, device, label="Unlearned (UL)")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ORIGINAL_CKPT = "/home/team2/Unlearning/ADVANCE/models/advance_trained_rerun_01.pth"
    SAVE_CKPT_PATH = "/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_ul.pth"
    FORGET_CLASS = "airport"
    
    run_UL_unlearning(
        original_ckpt=ORIGINAL_CKPT,
        save_ckpt_path=SAVE_CKPT_PATH,
        forget_class=FORGET_CLASS,
        device=device,
        lr=1e-5,
        epochs=15,
        batch_size=32,
        num_workers=4,
        patience=3
    )