"""
Compare All Unlearning Methods
===============================
Evaluates all unlearned models and generates a comparison table.
"""

import torch
import pandas as pd
import sys
import os

sys.path.insert(0, '/home/team2/Unlearning/ADVANCE')

from torch.utils.data import DataLoader
from src.dataset import get_forget_splits, get_retain_splits, ForgetDataset, RetainDataset
from src.labels import NUM_CLASSES

from unimodal_un.base_utils import load_model, evaluate_split
from mia_v2 import run_mia


def evaluate_model(model_path, model_name, forget_test_loader, retain_test_loader, 
                   full_forget_loader, full_retain_loader, device):
    """
    Evaluate a single model and return metrics.
    """
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")
    
    try:
        model = load_model(model_path, NUM_CLASSES, device)
    except Exception as e:
        print(f"  ✗ Failed to load: {e}")
        return None
    
    # Get accuracy matrix
    matrix = evaluate_split(model, forget_test_loader, retain_test_loader, device)
    
    # Get MIA results
    mia_results = run_mia(model, full_retain_loader, retain_test_loader,
                         full_forget_loader, device, label=model_name)
    
    return {
        'Model': model_name,
        'Forget Audio%': matrix[0, 0],
        'Forget Image%': matrix[0, 1],
        'Forget Fusion%': matrix[0, 2],
        'Retain Audio%': matrix[1, 0],
        'Retain Image%': matrix[1, 1],
        'Retain Fusion%': matrix[1, 2],
        'MIA Accuracy%': mia_results['mia_accuracy'],
        'MIA AUC%': mia_results['mia_auc'],
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Configuration
    FORGET_CLASS = "airport"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Models to compare
    MODELS = {
        'Original': '/home/team2/Unlearning/ADVANCE/models/advance_trained_rerun_01.pth',
        'NegGrad': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_neggrad.pth',
        'DTD': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_dtd.pth',
        'UL': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_ul.pth',
        'L-CODEC': '/home/team2/Unlearning/ADVANCE/models/unimodal_unlearn/advance_unlearned_lcodec.pth',
    
    }
    
    # Output directory
    OUTPUT_DIR = '/home/team2/Unlearning/ADVANCE/outputs/comparison'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("="*70)
    print("BUILDING DATASETS")
    print("="*70)
    
    # Get test sets
    _, _, forget_test = get_forget_splits(forget_class=FORGET_CLASS, seed=42)
    _, _, retain_test = get_retain_splits(forget_class=FORGET_CLASS, seed=42)
    
    forget_test_loader = DataLoader(forget_test, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
    retain_test_loader = DataLoader(retain_test, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
    
    # Full datasets for MIA
    full_forget = ForgetDataset(forget_class=FORGET_CLASS)
    full_retain = RetainDataset(forget_class=FORGET_CLASS)
    
    full_forget_loader = DataLoader(full_forget, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
    full_retain_loader = DataLoader(full_retain, batch_size=BATCH_SIZE,
                                    shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"\n  Forget test: {len(forget_test)}")
    print(f"  Retain test: {len(retain_test)}")
    
    # Evaluate all models
    results = []
    for model_name, model_path in MODELS.items():
        if not os.path.exists(model_path):
            print(f"\n✗ Skipping {model_name}: File not found at {model_path}")
            continue
        
        result = evaluate_model(
            model_path, model_name,
            forget_test_loader, retain_test_loader,
            full_forget_loader, full_retain_loader,
            device
        )
        
        if result is not None:
            results.append(result)
    
    # Create comparison table
    if not results:
        print("\n✗ No models were successfully evaluated!")
        return
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    csv_path = os.path.join(OUTPUT_DIR, 'unlearning_comparison.csv')
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"\n{'='*70}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*70}")
    
    # Print summary table
    print("\n" + "="*70)
    print("UNLEARNING METHODS COMPARISON")
    print("="*70)
    print(df.to_string(index=False, float_format=lambda x: f'{x:.2f}'))
    print("="*70)
    
    # Print key insights
    print("\n" + "="*70)
    print("KEY METRICS SUMMARY")
    print("="*70)
    
    print("\nForget Set Performance (Fusion) - Lower is better:")
    forget_fusion = df[['Model', 'Forget Fusion%']].sort_values('Forget Fusion%')
    for _, row in forget_fusion.iterrows():
        print(f"  {row['Model']:15} {row['Forget Fusion%']:6.2f}%")
    
    print("\nRetain Set Performance (Fusion) - Higher is better:")
    retain_fusion = df[['Model', 'Retain Fusion%']].sort_values('Retain Fusion%', ascending=False)
    for _, row in retain_fusion.iterrows():
        print(f"  {row['Model']:15} {row['Retain Fusion%']:6.2f}%")
    
    print("\nMIA Accuracy - Closer to 50% is better (indicates successful unlearning):")
    mia_acc = df[['Model', 'MIA Accuracy%']].copy()
    mia_acc['Distance from 50%'] = abs(mia_acc['MIA Accuracy%'] - 50.0)
    mia_acc = mia_acc.sort_values('Distance from 50%')
    for _, row in mia_acc.iterrows():
        print(f"  {row['Model']:15} {row['MIA Accuracy%']:6.2f}%  "
              f"(distance from 50%: {row['Distance from 50%']:.2f})")
    
    # Compute unlearning effectiveness score
    print("\n" + "="*70)
    print("UNLEARNING EFFECTIVENESS SCORE")
    print("="*70)
    print("Score = (100 - Forget%) + Retain% - |MIA% - 50|")
    print("Higher is better (balances forgetting + retention + privacy)")
    print()
    
    scores = []
    for _, row in df.iterrows():
        if row['Model'] == 'Original':
            continue
        
        forget_penalty = 100 - row['Forget Fusion%']
        retain_score = row['Retain Fusion%']
        mia_penalty = abs(row['MIA Accuracy%'] - 50)
        
        score = forget_penalty + retain_score - mia_penalty
        scores.append({
            'Model': row['Model'],
            'Score': score
        })
    
    score_df = pd.DataFrame(scores).sort_values('Score', ascending=False)
    for _, row in score_df.iterrows():
        print(f"  {row['Model']:15} {row['Score']:6.2f}")
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print(f"Full results: {csv_path}")
    print("="*70)


if __name__ == "__main__":
    main()