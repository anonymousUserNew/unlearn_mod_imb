"""
Membership Inference Attack (MIA) Evaluation
=============================================
Evaluates whether the unlearning was successful by checking if
forget samples can be distinguished from test samples.

A successful MIA means the model still "remembers" the forget set.
A failed MIA (≈50% accuracy) means successful unlearning.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm


@torch.no_grad()
def extract_confidence_scores(model, loader, device):
    """
    Extract confidence scores for MIA.
    For each sample, we use the max softmax probability as the confidence.

    Returns:
        confidences: np.array of max probabilities
    """
    model.eval()
    confidences = []

    for batch in tqdm(loader, desc="  Extracting confidences", leave=False):
        videos = batch['video'].to(device)
        specs  = batch['spectrogram'].to(device)

        out = model(videos, specs, return_intermediate=True)

        # Use fusion branch for MIA
        logits    = out['fusion_logits']
        probs     = torch.softmax(logits, dim=1)
        max_probs = probs.max(dim=1).values

        confidences.extend(max_probs.cpu().numpy())

    return np.array(confidences)


def run_mia(model, retain_loader, test_loader, forget_loader, device, label="Model"):
    """
    Run Membership Inference Attack.

    Methodology:
        - Member set: forget_loader (data model was trained on)
        - Non-member set: test_loader (held-out data)
        - Attack: classify based on confidence threshold
        - Success metric: MIA accuracy (higher = more memorization)

    Goal for unlearning: MIA accuracy ≈ 50% (random guessing)

    Args:
        model: DcaseMultimodalModel
        retain_loader: Retain set (for reference)
        test_loader: Non-member samples
        forget_loader: Member samples (forget set)
        device: torch device
        label: Label for printing
    """
    print(f"\n{'─'*70}")
    print(f"MIA Evaluation: {label}")
    print(f"{'─'*70}")

    # Extract confidences
    print("  Extracting forget set confidences (members)...")
    forget_conf = extract_confidence_scores(model, forget_loader, device)

    print("  Extracting test set confidences (non-members)...")
    test_conf = extract_confidence_scores(model, test_loader, device)

    # Create labels: 1 = member (forget), 0 = non-member (test)
    y_true = np.concatenate([
        np.ones(len(forget_conf)),   # forget = members
        np.zeros(len(test_conf))     # test = non-members
    ])

    # Scores: higher confidence = more likely to be member
    scores = np.concatenate([forget_conf, test_conf])

    # Simple threshold-based attack at median
    threshold = np.median(scores)
    y_pred    = (scores > threshold).astype(int)

    # Compute metrics
    mia_acc = accuracy_score(y_true, y_pred) * 100

    try:
        mia_auc = roc_auc_score(y_true, scores) * 100
    except Exception:
        mia_auc = 0.0

    # Statistics
    forget_mean = forget_conf.mean()
    forget_std  = forget_conf.std()
    test_mean   = test_conf.mean()
    test_std    = test_conf.std()

    print(f"\n  Confidence Statistics:")
    print(f"    Forget (member)    : {forget_mean:.4f} ± {forget_std:.4f}")
    print(f"    Test (non-member)  : {test_mean:.4f} ± {test_std:.4f}")
    print(f"    Separation         : {abs(forget_mean - test_mean):.4f}")

    print(f"\n  MIA Results:")
    print(f"    Accuracy : {mia_acc:.2f}%  (50% = random guess)")
    print(f"    AUC-ROC  : {mia_auc:.2f}%")

    # Interpretation
    if mia_acc > 60:
        print(f"    → High MIA accuracy: Model likely remembers forget set")
    elif mia_acc < 55:
        print(f"    → Low MIA accuracy: Effective unlearning")
    else:
        print(f"    → MIA accuracy near random: Moderate unlearning")

    print(f"{'─'*70}\n")

    return {
        'mia_accuracy':    mia_acc,
        'mia_auc':         mia_auc,
        'forget_conf_mean': forget_mean,
        'test_conf_mean':   test_mean,
    }