"""
NegGrad (Negative Gradient) Unlearning for Food101
==================================================
Gradient descent on retain set, gradient ascent on forget set.

Loss = retain_loss - lambda * forget_loss

References:
- Graves et al. (2021) "Amnesiac Machine Learning"
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import sys
import time

# Discover project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dataset import ForgetDataset, RetainDataset
from src.model_new_r import MultimodalFoodClassifier

# Import utilities
from base_utils import load_model, evaluate_split, print_accuracy_matrix, save_checkpoint
from mia_v2 import run_mia

# Constants
NUM_CLASSES = 101


def train_one_epoch_neggrad(model, retain_loader, forget_loader,
                            optimizer, device, epoch, forget_lambda=1.0):
    """
    NegGrad training epoch.
    
    Loss = retain_loss - lambda * forget_loss
    
    Args:
        forget_lambda: Weight for forget loss (default 1.0)
    """
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    # Cycle through both loaders together
    forget_iter = iter(forget_loader)
    
    total_loss = 0.0
    total_retain_loss = 0.0
    total_forget_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for retain_batch in tqdm(retain_loader, desc=f"  Epoch {epoch} [NegGrad]"):
        # Get retain batch
        r_imgs = retain_batch['image'].to(device)
        r_ids = retain_batch['input_ids'].to(device)
        r_mask = retain_batch['attention_mask'].to(device)
        r_labels = retain_batch['label'].to(device)
        
        # Get forget batch (cycle if needed)
        try:
            forget_batch = next(forget_iter)
        except StopIteration:
            forget_iter = iter(forget_loader)
            forget_batch = next(forget_iter)
        
        f_imgs = forget_batch['image'].to(device)
        f_ids = forget_batch['input_ids'].to(device)
        f_mask = forget_batch['attention_mask'].to(device)
        f_labels = forget_batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # ── Retain forward (gradient descent) ──
        out_r = model(r_imgs, r_ids, r_mask, return_intermediate=True)
        r_loss = (criterion(out_r['text_logits'], r_labels) +
                  criterion(out_r['image_logits'], r_labels) +
                  criterion(out_r['fusion_logits'], r_labels))
        
        # ── Forget forward (gradient ascent via negation) ──
        out_f = model(f_imgs, f_ids, f_mask, return_intermediate=True)
        f_loss = (criterion(out_f['text_logits'], f_labels) +
                  criterion(out_f['image_logits'], f_labels) +
                  criterion(out_f['fusion_logits'], f_labels))
        
        # ── Combined NegGrad loss ──
        loss = r_loss - forget_lambda * f_loss
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * r_labels.size(0)
        total_retain_loss += r_loss.item() * r_labels.size(0)
        total_forget_loss += f_loss.item() * f_labels.size(0)
        
        preds = out_r['fusion_logits'].argmax(dim=1)
        total_correct += (preds == r_labels).sum().item()
        total_samples += r_labels.size(0)
    
    n = total_samples
    print(f"\n  Epoch {epoch} Train Summary:")
    print(f"    Total Loss   : {total_loss/n:.4f}")
    print(f"    Retain Loss  : {total_retain_loss/n:.4f}  (minimized)")
    print(f"    Forget Loss  : {total_forget_loss/len(forget_loader.dataset):.4f}  (maximized)")
    print(f"    Retain Fusion Acc: {total_correct/n*100:.2f}%")


def run_neggrad_unlearning(original_ckpt, save_ckpt_path,
                           device, lr=1e-4, epochs=15, batch_size=32,
                           num_workers=4, patience=5, forget_lambda=1.0):
    """
    Run NegGrad unlearning on Food101 dataset.
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    forget_train = ForgetDataset()
    retain_train = RetainDataset()
    
    # Use full datasets for evaluation as well to ensure total forgetting/retention
    forget_test = forget_train
    retain_test = retain_train
    
    print(f"\n  Forget set size: {len(forget_train)}")
    print(f"  Retain set size: {len(retain_train)}")
    
    # Create dataloaders
    forget_train_loader = DataLoader(forget_train, batch_size=batch_size, 
                                     shuffle=True, num_workers=num_workers)
    forget_test_loader = DataLoader(forget_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    retain_train_loader = DataLoader(retain_train, batch_size=batch_size,
                                     shuffle=True, num_workers=num_workers)
    retain_test_loader = DataLoader(retain_test, batch_size=batch_size,
                                    shuffle=False, num_workers=num_workers)
    
    print("\n" + "="*70)
    print("LOADING ORIGINAL MODEL")
    print("="*70)
    
    model = load_model(original_ckpt, NUM_CLASSES, device)
    
    print("\nEvaluating BEFORE unlearning...")
    matrix_before = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_before, title="BEFORE UNLEARNING (Test Set)")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print("\n" + "="*70)
    print("NegGrad UNLEARNING")
    print(f"forget_lambda = {forget_lambda}")
    print("="*70)
    
    best_retain_acc = 0.0
    best_epoch = 0
    patience_count = 0
    
    for epoch in range(1, epochs + 1):
        print(f"\n{'-'*70}")
        print(f"Epoch [{epoch}/{epochs}]")
        print(f"{'-'*70}")
        
        train_one_epoch_neggrad(model, retain_train_loader, forget_train_loader,
                               optimizer, device, epoch, forget_lambda)
        scheduler.step()
        
        # Validate on retain set only (we want to preserve retain performance)
        matrix_val = evaluate_split(model, forget_test_loader, retain_test_loader, device)
        print_accuracy_matrix(matrix_val, title=f"EPOCH {epoch} — Test Set")
        
        retain_fusion_acc = matrix_val[1, 2]  # Retain, fusion branch
        
        if retain_fusion_acc > best_retain_acc:
            best_retain_acc = retain_fusion_acc
            best_epoch = epoch
            patience_count = 0
            save_checkpoint(model, save_ckpt_path, epoch, retain_fusion_acc)
            print(f"✓ New best (epoch {epoch}, retain_fusion={retain_fusion_acc:.2f}%)")
        else:
            patience_count += 1
            print(f"  No improvement. Patience {patience_count}/{patience}")
            if patience_count >= patience:
                print(f"\n  Early stopping at epoch {epoch}.")
                print(f"  Best: epoch {best_epoch}, retain_fusion={best_retain_acc:.2f}%")
                break
    
    print("\n" + "="*70)
    print(f"TRAINING COMPLETED!")
    print(f"Best epoch: {best_epoch}  |  Best retain acc: {best_retain_acc:.2f}%")
    print("="*70)
    
    # Load best checkpoint for final evaluation
    print("\nLoading best checkpoint for final evaluation...")
    model = load_model(save_ckpt_path, NUM_CLASSES, device)
    
    matrix_after = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_after, title="AFTER UNLEARNING (Test Set)")
    
    # MIA Evaluation
    print("\n" + "="*70)
    print("MIA EVALUATION")
    print("="*70)
    
    print("\n[Original Model]")
    orig_model = load_model(original_ckpt, NUM_CLASSES, device)
    
    run_mia(orig_model, retain_train_loader, retain_test_loader,
            forget_train_loader, device, label="Original")
    
    print("\n[Unlearned Model - NegGrad]")
    run_mia(model, retain_train_loader, retain_test_loader,
            forget_train_loader, device, label="Unlearned (NegGrad)")
    
    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Configuration
    ORIGINAL_CKPT = "/home/team2/Unlearning/Food101-files/models/best_multimodal_food101.pth"
    SAVE_CKPT_PATH = "/home/team2/Unlearning/Food101-files/models/unimodal_unlearn/food101_unlearned_neggrad.pth"
    
    _t_start = time.perf_counter()
    run_neggrad_unlearning(
        original_ckpt=ORIGINAL_CKPT,
        save_ckpt_path=SAVE_CKPT_PATH,
        device=device,
        lr=1e-5,
        epochs=15,
        batch_size=16,
        num_workers=4,
        patience=3,
        forget_lambda=1.0
    )
    _t_end = time.perf_counter()
    _elapsed = _t_end - _t_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Runtime (train.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
    print(f"{'='*60}")