import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model import AdvanceMultimodalModel
from src.labels import NUM_CLASSES, LABEL_MAP, ADVANCE_CLASSES
from src.dataset import FORGET_CLASS

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
FORGET_IDX = LABEL_MAP[FORGET_CLASS]


# =============================================================================
# MIA Attack
# =============================================================================
def calculate_mia(forget_probs, test_probs):
    """
    Logistic regression-based MIA attack.
    
    Args:
        forget_probs: array of max probabilities from forget set
        test_probs:   array of max probabilities from retain set
    
    Returns:
        MIA accuracy (%) — higher = model leaks membership info
        For unlearning: lower is better (should approach 50%)
    """
    min_len = min(len(forget_probs), len(test_probs))
    if min_len == 0:
        return 50.0
    
    f_probs = np.random.choice(forget_probs, min_len, replace=False)
    t_probs = np.random.choice(test_probs, min_len, replace=False)
    
    X = np.concatenate([f_probs, t_probs]).reshape(-1, 1)
    y = np.concatenate([np.ones(min_len), np.zeros(min_len)])
    
    clf = LogisticRegression(solver='lbfgs', max_iter=1000)
    clf.fit(X, y)
    preds = clf.predict(X)
    
    return accuracy_score(y, preds) * 100.0


# =============================================================================
# Inference
# =============================================================================
def collect_outputs(data_loader, model, device):
    """Returns (probs, preds, labels) all as numpy arrays."""
    all_probs, all_preds, all_labels = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            img    = batch["image"].to(device)
            spec   = batch["spectrogram"].to(device)
            labels = batch["label"].to(device)

            output = model(img, spec)
            if isinstance(output, dict):
                output = output["fusion_logits"]

            probs = F.softmax(output, dim=-1)
            preds = probs.argmax(dim=-1)

            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    return (
        torch.cat(all_probs).numpy(),
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
    )


# =============================================================================
# Metrics
# =============================================================================
def entropy(probs):
    """Shannon entropy per sample. Shape: (N,)"""
    log_p = np.log(np.maximum(probs, 1e-30))
    return -np.sum(probs * log_p, axis=1)


def evaluate_class_unlearning(forget_loader, retain_loader, model, device):
    """
    Core evaluation for class-wise unlearning.

    Forget set metrics (lower forget accuracy / confidence = better unlearning):
      - forget_acc          : fraction of forget samples still predicted as forget class
      - forget_confidence   : avg softmax score on the true forget label
      - forget_entropy      : avg entropy of predictions on forget samples (higher = more confused)

    Retain set metrics (should stay high):
      - retain_acc          : accuracy on retain samples (excluding forget class)

    MIA Attack score (should approach 50% for good unlearning):
      - mia_attack_acc      : logistic regression classifier accuracy distinguishing forget vs retain
    """
    # ── forget set ────────────────────────────────────────────────────────────
    f_probs, f_preds, f_labels = collect_outputs(forget_loader, model, device)

    forget_acc        = np.mean(f_preds == FORGET_IDX)          # still predicts forget class
    forget_confidence = np.mean(f_probs[:, FORGET_IDX])         # avg score on forget label
    forget_entropy_   = np.mean(entropy(f_probs))               # avg entropy on forget samples
    forget_max_probs  = np.max(f_probs, axis=1)                 # max softmax per sample

    # How often does it predict the correct forget label vs scatter to other classes
    pred_distribution = np.bincount(f_preds, minlength=NUM_CLASSES) / len(f_preds)

    # ── retain set ────────────────────────────────────────────────────────────
    r_probs, r_preds, r_labels = collect_outputs(retain_loader, model, device)
    retain_acc = np.mean(r_preds == r_labels)
    retain_max_probs = np.max(r_probs, axis=1)                 # max softmax per sample

    # ── MIA Attack ────────────────────────────────────────────────────────────
    # Train logistic regression to distinguish forget samples from retain samples
    # using max probability as feature. Lower is better (50% = random guessing)
    mia_attack_acc = calculate_mia(forget_max_probs, retain_max_probs)

    return {
        "forget_acc":         forget_acc,
        "forget_confidence":  forget_confidence,
        "forget_entropy":     forget_entropy_,
        "retain_acc":         retain_acc,
        "mia_attack_acc":     mia_attack_acc,
        "pred_distribution":  pred_distribution,
    }


def print_results(name, metrics):
    print(f"\n{'='*55}")
    print(f"  Model: {name}")
    print(f"{'='*55}")
    print(f"  Forget class          : {FORGET_CLASS} (idx {FORGET_IDX})")
    print(f"  {'─'*50}")
    print(f"  [Forget Set — lower is better for unlearning]")
    print(f"    Forget class accuracy    : {metrics['forget_acc']:.4f}   (ideal: 0.0)")
    print(f"    Forget class confidence  : {metrics['forget_confidence']:.4f}   (ideal: 0.0)")
    print(f"    Forget sample entropy    : {metrics['forget_entropy']:.4f}   (ideal: high)")
    print(f"  {'─'*50}")
    print(f"  [Retain Set — higher is better]")
    print(f"    Retain accuracy          : {metrics['retain_acc']:.4f}   (ideal: 1.0)")
    print(f"  {'─'*50}")
    print(f"  [MIA Attack — lower is better for unlearning]")
    print(f"    MIA attack accuracy      : {metrics['mia_attack_acc']:.2f}%   (ideal: ~50%)")
    print(f"  {'─'*50}")
    print(f"  [Prediction distribution on forget samples]")
    for cls_idx, frac in enumerate(metrics['pred_distribution']):
        if frac > 0.01:   # only show classes with >1% predictions
            marker = " ← forget class" if cls_idx == FORGET_IDX else ""
            print(f"    {ADVANCE_CLASSES[cls_idx]:<25}: {frac:.3f}{marker}")
    print(f"{'='*55}")


# =============================================================================
# Main
# =============================================================================
def main():
    print("Loading datasets...")
    forget_train, forget_val, forget_test = get_forget_splits(seed=42)
    retain_train, retain_val, retain_test = get_retain_splits(seed=42)

    # Use forget_train as the probe set (what the model was trained on / should forget)
    forget_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # Use retain_test as the retain probe (held-out, not seen during unlearning)
    retain_loader = DataLoader(retain_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    models_to_eval = {
        "trained":        "/home/team2/Unlearning/ADVANCE/models/advance_trained_rerun_01.pth",
        "unlearned_ours": "/home/team2/Unlearning/ADVANCE/models/advance_unlearned_4loss_01_rerun.pth",
        "multidelete":    "/home/team2/Unlearning/ADVANCE/models/advance_unlearned_embed_rerun_01.pth",
    }

    model = AdvanceMultimodalModel(num_classes=NUM_CLASSES)
    model.to(DEVICE)

    for name, path in models_to_eval.items():
        try:
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            metrics = evaluate_class_unlearning(forget_loader, retain_loader, model, DEVICE)
            print_results(name, metrics)
        except Exception as e:
            print(f"Error evaluating {name}: {e}")


if __name__ == "__main__":
    main()