"""
Unified Comparison Script for All Unlearning Methods
=====================================================
Evaluates all unlearned models on FULL forget/retain datasets
using the SAME evaluation methodology.

Generates a comprehensive comparison table with:
- Audio, Image, Fusion accuracies
- Forget set performance
- Retain set performance
- All 6 metrics for fair comparison
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.insert(0, '/home/team2/Unlearning/ADVANCE')

from src.dataset import ForgetDataset, RetainDataset
from src.model import AdvanceMultimodalModel
from src.labels import NUM_CLASSES, ADVANCE_CLASSES


def evaluate_single_branch(model, loader, device, branch='fusion'):
    """
    Evaluate a single branch (audio/image/fusion) on a dataset.
    
    Returns:
        accuracy: float (percentage)
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Evaluating {branch}", leave=False):
            images = batch['image'].to(device)
            specs = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            
            out = model(images, specs, return_intermediate=True)
            
            if branch == 'audio':
                logits = out['audio_logits']
            elif branch == 'image':
                logits = out['image_logits']
            else:  # fusion
                logits = out['fusion_logits']
            
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    return accuracy


def evaluate_model(model_path, model_name, forget_loader, retain_loader, device):
    """
    Evaluate a model on both forget and retain sets, all 3 branches.
    
    Returns:
        dict with 6 metrics: forget_audio, forget_image, forget_fusion,
                            retain_audio, retain_image, retain_fusion
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    
    # Load model
    try:
        model = AdvanceMultimodalModel(num_classes=NUM_CLASSES)
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return None
    
    # Evaluate all branches on forget set
    print("\n  Forget Set:")
    forget_audio = evaluate_single_branch(model, forget_loader, device, 'audio')
    forget_image = evaluate_single_branch(model, forget_loader, device, 'image')
    forget_fusion = evaluate_single_branch(model, forget_loader, device, 'fusion')
    
    print(f"    Audio : {forget_audio:.2f}%")
    print(f"    Image : {forget_image:.2f}%")
    print(f"    Fusion: {forget_fusion:.2f}%")
    
    # Evaluate all branches on retain set
    print("\n  Retain Set:")
    retain_audio = evaluate_single_branch(model, retain_loader, device, 'audio')
    retain_image = evaluate_single_branch(model, retain_loader, device, 'image')
    retain_fusion = evaluate_single_branch(model, retain_loader, device, 'fusion')
    
    print(f"    Audio : {retain_audio:.2f}%")
    print(f"    Image : {retain_image:.2f}%")
    print(f"    Fusion: {retain_fusion:.2f}%")
    
    return {
        'Method': model_name,
        'Forget Audio%': forget_audio,
        'Forget Image%': forget_image,
        'Forget Fusion%': forget_fusion,
        'Retain Audio%': retain_audio,
        'Retain Image%': retain_image,
        'Retain Fusion%': retain_fusion,
    }


def main():
    # Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FORGET_CLASS = "airport"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    print(f"Device: {DEVICE}")
    print(f"Forget class: {FORGET_CLASS}\n")
    
    # Define models to compare
    # EDIT THESE PATHS TO MATCH YOUR SETUP
    MODELS = {
        'Original (Baseline)': '/home/team2/Unlearning/ADVANCE/models/advance_trained_rerun_01.pth',
        'Your Multimodal Method': '/home/team2/Unlearning/ADVANCE/models/advance_unlearned_4loss_01_rerun.pth',
        'NegGrad': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_neggrad.pth',
        'DTD': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_dtd.pth',
        'UL': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_ul.pth',
        'L-CODEC': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_lcodec.pth',
    }
    
    # Output directory
    OUTPUT_DIR = '/home/team2/Unlearning/ADVANCE/outputs/unified_comparison'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load FULL datasets (not splits)
    print("="*70)
    print("LOADING DATASETS (FULL)")
    print("="*70)
    
    full_forget = ForgetDataset(forget_class=FORGET_CLASS)
    full_retain = RetainDataset(forget_class=FORGET_CLASS)
    
    print(f"  Forget samples: {len(full_forget)}")
    print(f"  Retain samples: {len(full_retain)}")
    
    forget_loader = DataLoader(full_forget, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    retain_loader = DataLoader(full_retain, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=NUM_WORKERS)
    
    # Evaluate all models
    results = []
    
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n✗ Skipping {model_name}: File not found")
            print(f"  Expected at: {model_path}")
            continue
        
        result = evaluate_model(model_path, model_name, 
                              forget_loader, retain_loader, DEVICE)
        
        if result is not None:
            results.append(result)
    
    # Check if we have results
    if not results:
        print("\n✗ No models were successfully evaluated!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder columns for clarity
    column_order = [
        'Method',
        'Forget Audio%', 'Forget Image%', 'Forget Fusion%',
        'Retain Audio%', 'Retain Image%', 'Retain Fusion%',
    ]
    df = df[column_order]
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'all_methods_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.2f')
    
    # Print results
    print("\n" + "="*70)
    print("UNIFIED COMPARISON - ALL METHODS")
    print("="*70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*70)
    
    # Print LaTeX table (for paper)
    print("\n" + "="*70)
    print("LaTeX TABLE (Copy-paste into your paper)")
    print("="*70)
    
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Comparison of Unlearning Methods on ADVANCE Dataset}")
    print("\\begin{tabular}{l|ccc|ccc}")
    print("\\hline")
    print("\\multirow{2}{*}{Method} & \\multicolumn{3}{c|}{Forget Set} & \\multicolumn{3}{c}{Retain Set} \\\\")
    print(" & Audio & Image & Fusion & Audio & Image & Fusion \\\\")
    print("\\hline")
    
    for _, row in df.iterrows():
        print(f"{row['Method']:25} & {row['Forget Audio%']:5.2f} & {row['Forget Image%']:5.2f} & {row['Forget Fusion%']:5.2f} & {row['Retain Audio%']:5.2f} & {row['Retain Image%']:5.2f} & {row['Retain Fusion%']:5.2f} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    print("="*70)
    
    # Create visualization-friendly summary
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    
    # Best forgetting (lowest forget fusion %)
    best_forget = df.loc[df['Forget Fusion%'].idxmin()]
    print(f"\nBest Forgetting (Fusion): {best_forget['Method']}")
    print(f"  Forget Fusion%: {best_forget['Forget Fusion%']:.2f}%")
    
    # Best retention (highest retain fusion %)
    best_retain = df.loc[df['Retain Fusion%'].idxmax()]
    print(f"\nBest Retention (Fusion): {best_retain['Method']}")
    print(f"  Retain Fusion%: {best_retain['Retain Fusion%']:.2f}%")
    
    # Compute effectiveness score (balance of forgetting + retention)
    df_score = df.copy()
    df_score['Effectiveness'] = (100 - df_score['Forget Fusion%']) + df_score['Retain Fusion%']
    best_overall = df_score.loc[df_score['Effectiveness'].idxmax()]
    
    print(f"\nBest Overall (Forget+Retain Balance): {best_overall['Method']}")
    print(f"  Effectiveness Score: {best_overall['Effectiveness']:.2f}")
    print(f"  (Score = (100 - Forget%) + Retain%)")
    
    print("\n" + "="*70)
    print(f"Results saved to: {csv_path}")
    print("="*70)


if __name__ == "__main__":
    main()