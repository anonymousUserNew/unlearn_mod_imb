"""
L-CODEC (Conditional Independence) Unlearning for CREMA-D
==========================================================
One-shot perturbation based on Fisher information matrices.
No training loop required.

References:
- Mehta et al. (2022) "Deep Unlearning via Randomized Conditionally Independent Hessians"
"""

import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model import CremaMultimodalModel
from src.labels import NUM_CLASSES

from base_utils import load_model, evaluate_split, print_accuracy_matrix, save_checkpoint
from mia_v2 import run_mia


def compute_diagonal_fisher(model, loader, device, desc=""):
    """
    Compute diagonal Fisher information matrix.
    Fisher_i = (1/N) * sum((dL/dw_i)^2)
    
    Returns:
        fisher: dict of param_name -> tensor (same shape as param)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    fisher = {}
    # Initialize
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher[name] = torch.zeros_like(param.data, device=device)
    
    total_samples = 0
    
    for batch in tqdm(loader, desc=f"  Computing Fisher [{desc}]"):
        vids = batch['video'].to(device)
        specs = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        
        model.zero_grad()
        
        out = model(vids, specs, return_intermediate=True)
        
        loss = (criterion(out['audio_logits'], labels) +
                criterion(out['video_logits'], labels) +
                criterion(out['fusion_logits'], labels))
        
        loss.backward()
        
        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher[name] += param.grad.data.pow(2).to(device) * vids.size(0)
        
        total_samples += vids.size(0)
    
    # Normalize
    for name in fisher:
        fisher[name] /= total_samples
    
    model.zero_grad()
    
    print(f"  Fisher computed over {total_samples} samples [{desc}]")
    return fisher


def compute_perturbation_scale(forget_fisher, retain_fisher, 
                                noise_multiplier=1.0, eps=1e-8):
    """
    Compute per-weight perturbation scale using conditional independence.
    
    sigma_i = sqrt(forget_fisher_i / (retain_fisher_i + forget_fisher_i + eps)^2)
    
    Returns:
        sigma_dict: dict of param_name -> perturbation std
    """
    sigma_dict = {}
    
    for name in forget_fisher:
        if name not in retain_fisher:
            continue
        
        f_fish = forget_fisher[name]
        r_fish = retain_fisher[name]
        
        # Conditional independence formula
        denom = (r_fish + f_fish + eps).pow(2)
        sigma = torch.sqrt(f_fish / denom + eps) * noise_multiplier
        sigma = torch.clamp(sigma, max=0.1)
        
        # Handle NaNs
        if torch.isnan(sigma).any():
            sigma = torch.nan_to_num(sigma, nan=0.0, posinf=1e3, neginf=-1e3)
        
        sigma_dict[name] = sigma
    
    # Diagnostics
    all_sigma = torch.cat([s.flatten().cpu() for s in sigma_dict.values()])
    print(f"\n  Perturbation Scale Summary:")
    print(f"    Min  : {all_sigma.min().item():.6f}")
    print(f"    Max  : {all_sigma.max().item():.6f}")
    print(f"    Mean : {all_sigma.mean().item():.6f}")
    print(f"    Noise multiplier: {noise_multiplier}")
    
    return sigma_dict


def apply_lcodec_perturbation(model, sigma_dict, device):
    """
    Apply one-shot weight perturbation.
    This IS the unlearning step - no training loop needed.
    """
    model.eval()
    
    total_perturbed = 0
    total_params = 0
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            total_params += param.numel()
            if name in sigma_dict:
                sigma = sigma_dict[name].to(param.device)
                noise = torch.randn_like(param.data, device=param.device) * sigma
                param.data.add_(noise)
                total_perturbed += param.numel()
    
    print(f"\n  L-CODEC perturbation applied:")
    print(f"    Parameters perturbed: {total_perturbed}/{total_params} "
          f"({100*total_perturbed/total_params:.1f}%)")
    print(f"    One-shot unlearning complete")


def run_L_CODEC_unlearning(original_ckpt, save_ckpt_path, forget_class,
                           device, batch_size=32, num_workers=4,
                           noise_multiplier=1.0):
    """
    Run L-CODEC unlearning on CREMA-D dataset.
    
    This is a one-shot method - no training loop.
    """
    print("\n" + "="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    # Get forget/retain train splits for Fisher (mimicking the training distribution)
    forget_train, _, forget_test = get_forget_splits(forget_class=forget_class, seed=42)
    retain_train, _, retain_test = get_retain_splits(forget_class=forget_class, seed=42)
    
    print(f"\n  Forget class: {forget_class}")
    print(f"  Retain train samples: {len(retain_train)}")
    print(f"  Forget train samples: {len(forget_train)}")
    
    # Create loaders for Fisher computation
    retain_loader = DataLoader(retain_train, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
    forget_loader = DataLoader(forget_train, batch_size=batch_size,
                               shuffle=False, num_workers=num_workers)
    
    # Create loaders for evaluation
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
    
    print("\n" + "="*70)
    print("L-CODEC UNLEARNING (One-Shot)")
    print(f"noise_multiplier = {noise_multiplier}")
    print("="*70)
    
    print("\nStep 1: Computing Fisher matrices...")
    forget_fisher = compute_diagonal_fisher(model, forget_loader, device, desc="Forget")
    retain_fisher = compute_diagonal_fisher(model, retain_loader, device, desc="Retain")
    
    print("\nStep 2: Computing perturbation scales...")
    sigma_dict = compute_perturbation_scale(forget_fisher, retain_fisher, 
                                           noise_multiplier=noise_multiplier)
    
    print("\nStep 3: Applying one-shot perturbation...")
    apply_lcodec_perturbation(model, sigma_dict, device)
    
    print("\n  L-CODEC complete. Saving model...")
    save_checkpoint(model, save_ckpt_path, epoch=0, val_acc=0.0)
    
    print("\nEvaluating AFTER unlearning...")
    matrix_after = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    print_accuracy_matrix(matrix_after, title="AFTER UNLEARNING")
    
    # MIA
    print("\n" + "="*70)
    print("MIA EVALUATION")
    print("="*70)
    
    print("\n[Original Model]")
    orig_model = load_model(original_ckpt, NUM_CLASSES, device)
    run_mia(orig_model, retain_loader, retain_test_loader,
            forget_loader, device, label="Original")
    
    print("\n[Unlearned Model - L-CODEC]")
    run_mia(model, retain_loader, retain_test_loader,
            forget_loader, device, label="Unlearned (L-CODEC)")


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    ORIGINAL_CKPT = "/home/team2/Unlearning/crema-d-mirror/models/crema_trained_05.pth"
    SAVE_CKPT_PATH = "/home/team2/Unlearning/crema-d-mirror/models/unimodal_unlearn/crema_unlearned_lcodec.pth"
    FORGET_CLASS = "HAP"
    
    _t_start = time.perf_counter()
    run_L_CODEC_unlearning(
        original_ckpt=ORIGINAL_CKPT,
        save_ckpt_path=SAVE_CKPT_PATH,
        forget_class=FORGET_CLASS,
        device=device,
        batch_size=32,
        num_workers=4,
        noise_multiplier=1.0
    )
    _t_end = time.perf_counter()
    _elapsed = _t_end - _t_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    print(f"\n{'='*60}")
    print(f"  Runtime (dtd_advance.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
    print(f"{'='*60}")