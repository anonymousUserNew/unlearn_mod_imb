"""
DTD (Descent-to-Delete) Unlearning for DCASE
===============================================
Train on retain set with Gaussian noise injection for privacy.

References:
- Neel et al. (2021) "Descent-to-Delete"
"""

import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.insert(0, '/home/team2/Unlearning/Dcase')

from src.dataset import get_forget_splits, get_retain_splits
from src.model import DcaseMultimodalModel
from src.labels import NUM_CLASSES

from base_utils import load_model, evaluate_split, print_accuracy_matrix, save_checkpoint
from mia_v2 import run_mia


def inject_gradient_noise(model, sigma, device):
    """
    Inject calibrated Gaussian noise into gradients.
    Called after backward(), before optimizer.step().
    """
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.randn_like(param.grad) * sigma
            param.grad.add_(noise)


def train_one_epoch_DTD(model, retain_loader, optimizer, device, epoch,
                        sigma, grad_clip=1.0):
    """
    DTD training epoch.
    Train on retain set only, with noisy gradients.
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in tqdm(retain_loader, desc=f"  Epoch {epoch} [DTD]"):
        vids = batch['video'].to(device)
        specs = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        out = model(vids, specs, return_intermediate=True)
        
        loss = (criterion(out['audio_logits'], labels) +
                criterion(out['video_logits'], labels) +
                criterion(out['fusion_logits'], labels))
        
        loss.backward()
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        # Inject Gaussian noise
        inject_gradient_noise(model, sigma, device)
        
        optimizer.step()
        
        total_loss += loss.item() * labels.size(0)
        preds = out['fusion_logits'].argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
    
    n = total_samples
    print(f"\n  Epoch {epoch} Train Summary:")
    print(f"    Retain CE Loss     : {total_loss/n:.4f}")
    print(f"    Retain Fusion Acc  : {total_correct/n*100:.2f}%")
    print(f"    Gradient noise σ   : {sigma:.6f}")
    print(f"    Gradient clip norm : {grad_clip:.4f}")


def run_DTD_unlearning(original_ckpt, save_ckpt_path, forget_class,
                       device, lr=1e-4, epochs=15, batch_size=32,
                       num_workers=4, patience=5, noise_multiplier=0.01,
                       grad_clip=1.0):
    """
    Run DTD unlearning on DCASE dataset.
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    forget_train, forget_val, forget_test = get_forget_splits(forget_class=forget_class, seed=42)
    retain_train, retain_val, retain_test = get_retain_splits(forget_class=forget_class, seed=42)
    
    print(f"\n  Forget class: {forget_class}")
    print(f"  Retain train: {len(retain_train)}")
    print(f"  Retain test : {len(retain_test)}")
    
    retain_train_loader = DataLoader(retain_train, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    # Train sets for MIA
    forget_train_loader = DataLoader(forget_train, batch_size=batch_size,
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
    print("DTD UNLEARNING")
    print(f"noise_multiplier = {noise_multiplier}")
    print(f"grad_clip = {grad_clip}")
    print("="*70)
    
    best_retain_acc = 0.0
    best_epoch = 0
    patience_count = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'-'*70}")
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"{'-'*70}")
        
        sigma = noise_multiplier * lr
        train_one_epoch_DTD(model, retain_train_loader, optimizer, device,
                           epoch, sigma, grad_clip)
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
    
    print("\n[Unlearned Model - DTD]")
    run_mia(model, retain_train_loader, retain_test_loader,
            forget_train_loader, device, label="Unlearned (DTD)")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ORIGINAL_CKPT = "/home/team2/Unlearning/Dcase/models_rte/dcase_trained.pth"
    SAVE_CKPT_PATH = "/home/team2/Unlearning/Dcase/models_rte/unimodal_un/dcase_unlearned_dtd.pth"
    FORGET_CLASS = "bus"

    _t_start = time.perf_counter()
    run_DTD_unlearning(
        original_ckpt=ORIGINAL_CKPT,
        save_ckpt_path=SAVE_CKPT_PATH,
        forget_class=FORGET_CLASS,
        device=device,
        lr=1e-5,
        epochs=15,
        batch_size=32,
        num_workers=4,
        patience=3,
        noise_multiplier=0.01,
        grad_clip=1.0
    )
    _t_end = time.perf_counter()
    _elapsed = _t_end - _t_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Runtime (dtd_advance.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
    print(f"{'='*60}")