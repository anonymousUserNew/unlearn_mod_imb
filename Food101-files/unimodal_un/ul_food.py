"""
UL (Uncertainty Learning) Unlearning for Food101
=================================================
Push model toward maximum uncertainty on forget samples.
Uses KL divergence to uniform distribution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import time
import sys

# Discover project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import ForgetDataset, RetainDataset
from src.model_new_r import MultimodalFoodClassifier

from base_utils import load_model, evaluate_split, print_accuracy_matrix, save_checkpoint
from mia_v2 import run_mia

# Constants
NUM_CLASSES = 101


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
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        # Note: labels not used
        
        optimizer.zero_grad()
        
        out = model(imgs, ids, mask, return_intermediate=True)
        
        # Uncertainty loss on all three branches
        loss = (uncertainty_loss(out['text_logits'], NUM_CLASSES) +
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


def run_UL_unlearning(original_ckpt, save_ckpt_path,
                      device, lr=1e-4, epochs=15, batch_size=32,
                      num_workers=4, patience=5):
    """
    Run UL unlearning on Food101 dataset.
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    forget_train = ForgetDataset()
    retain_train = RetainDataset()
    
    forget_test = forget_train
    retain_test = retain_train
    
    print(f"  Forget set size: {len(forget_train)}")
    print(f"  Retain set size : {len(retain_train)}")
    
    forget_train_loader = DataLoader(forget_train, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    # Train sets for MIA
    retain_train_loader = DataLoader(retain_train, batch_size=batch_size,
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
    
    print("\n[Original Model]")
    orig_model = load_model(original_ckpt, NUM_CLASSES, device)
    run_mia(orig_model, retain_train_loader, retain_test_loader,
            forget_train_loader, device, label="Original")
    
    print("\n[Unlearned Model - UL]")
    run_mia(model, retain_train_loader, retain_test_loader,
            forget_train_loader, device, label="Unlearned (UL)")


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ORIGINAL_CKPT = "/home/team2/Unlearning/Food101-files/models/best_multimodal_food101.pth"
    SAVE_CKPT_PATH = "/home/team2/Unlearning/Food101-files/models/unimodal_unlearn/food101_unlearned_ul.pth"
    
    _t_start = time.perf_counter()
    run_UL_unlearning(
        original_ckpt=ORIGINAL_CKPT,
        save_ckpt_path=SAVE_CKPT_PATH,
        device=device,
        lr=1e-5,
        epochs=15,
        batch_size=16,
        num_workers=4,
        patience=3
    )
    _t_end = time.perf_counter()
    _elapsed = _t_end - _t_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Runtime (train.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
    print(f"{'='*60}")