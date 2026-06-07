"""
Membership Inference Attack Evaluation
========================================
Evaluates unlearning effectiveness by checking whether forget samples
look like non-members to the unlearned model.

Three populations:
  - forget_train  : samples the model was asked to forget (were members)
  - retain_train  : samples that should remain members
  - test          : true non-members (never seen during training)

Good unlearning: forget_train scores on unlearned model ≈ test scores
Bad unlearning : forget_train scores on unlearned model ≈ retain_train scores
"""

import os
import sys
import json
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import DcaseMultimodalModel
from src.labels  import NUM_CLASSES

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BATCH_SIZE = 16
C          = NUM_CLASSES

TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "dcase_trained.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "dcase_unlearned_embed.pth")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "inference_attack")
os.makedirs(OUTPUT_DIR, exist_ok=True)

METRICS = ["confidence", "max_prob", "entropy", "margin"]

# ──────────────────────────────────────────────────────────────────────────────
# DATA  — three distinct populations
# ──────────────────────────────────────────────────────────────────────────────
forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Use combined test splits as the "true non-member" pool
test_set = torch.utils.data.ConcatDataset([forget_test, retain_test])

forget_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=False,
                           drop_last=False, num_workers=4)
retain_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=False,
                           drop_last=False, num_workers=4)
test_loader   = DataLoader(test_set,    batch_size=BATCH_SIZE, shuffle=False,
                           drop_last=False, num_workers=4)

print(f"Populations  →  forget_train: {len(forget_train)} | "
      f"retain_train: {len(retain_train)} | test: {len(test_set)}")

# ──────────────────────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────────────────────
def load_model(path):
    m = DcaseMultimodalModel(num_classes=C).to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

model_trained   = load_model(TRAINED_MODEL_PATH)
model_unlearned = load_model(UNLEARNED_MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# SCORING
# ──────────────────────────────────────────────────────────────────────────────
def score_loader(model, loader, metric):
    """
    Returns a 1-D numpy array of membership scores for every sample in loader.
    Higher score = model thinks this sample is a member.
    """
    all_scores = []
    with torch.no_grad():
        for batch in loader:
            video  = batch["video"].to(DEVICE)
            spec   = batch["spectrogram"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(video, spec, return_intermediate=False)
            probs  = F.softmax(logits, dim=1)

            if metric == "confidence":
                # Probability assigned to the true class
                s = probs[torch.arange(len(labels)), labels].cpu().numpy()

            elif metric == "max_prob":
                s = probs.max(dim=1).values.cpu().numpy()

            elif metric == "entropy":
                # Negate: high entropy → low confidence → likely non-member
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1)
                s = -entropy.cpu().numpy()

            elif metric == "margin":
                top2 = torch.topk(probs, 2, dim=1).values
                s = (top2[:, 0] - top2[:, 1]).cpu().numpy()

            else:
                raise ValueError(f"Unknown metric: {metric}")

            all_scores.append(s)

    return np.concatenate(all_scores)


# ──────────────────────────────────────────────────────────────────────────────
# COLLECT SCORES for all three populations × two models × all metrics
# ──────────────────────────────────────────────────────────────────────────────
populations = {
    "forget" : forget_loader,
    "retain" : retain_loader,
    "test"   : test_loader,
}

print("\nCollecting scores (this may take a while)...")
scores = {}   # scores[model_name][pop][metric]

for model_name, model in [("trained", model_trained), ("unlearned", model_unlearned)]:
    scores[model_name] = {}
    for pop, loader in populations.items():
        scores[model_name][pop] = {}
        for metric in METRICS:
            scores[model_name][pop][metric] = score_loader(model, loader, metric)
            print(f"  {model_name:10s} | {pop:7s} | {metric:12s} | "
                  f"mean={scores[model_name][pop][metric].mean():.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────────────────────
# Standard MIA: member = retain_train, non-member = test
# Unlearning MIA: member = forget_train, non-member = test
#   → want AUC ≈ 0.5 on unlearned model for forget set

def mia_auc(member_scores, nonmember_scores):
    y     = np.concatenate([np.ones(len(member_scores)),
                            np.zeros(len(nonmember_scores))])
    s     = np.concatenate([member_scores, nonmember_scores])
    return float(roc_auc_score(y, s))

results = {}

print(f"\n{'='*80}")
print("MEMBERSHIP INFERENCE ATTACK — RESULTS")
print(f"{'='*80}")

for metric in METRICS:
    results[metric] = {}

    # ── Standard MIA (retain vs test) ────────────────────────────────────────
    auc_std_trained   = mia_auc(scores["trained"]["retain"][metric],
                                 scores["trained"]["test"][metric])
    auc_std_unlearned = mia_auc(scores["unlearned"]["retain"][metric],
                                 scores["unlearned"]["test"][metric])

    # ── Forget MIA (forget vs test) ───────────────────────────────────────────
    # On the TRAINED model: forget samples should look like members → AUC high
    # On the UNLEARNED model: forget samples should look like non-members → AUC ≈ 0.5
    auc_forget_trained   = mia_auc(scores["trained"]["forget"][metric],
                                    scores["trained"]["test"][metric])
    auc_forget_unlearned = mia_auc(scores["unlearned"]["forget"][metric],
                                    scores["unlearned"]["test"][metric])

    # ── Mean scores per population ────────────────────────────────────────────
    mean = {
        model: {pop: float(scores[model][pop][metric].mean())
                for pop in populations}
        for model in ("trained", "unlearned")
    }

    results[metric] = {
        "auc_standard_trained"    : auc_std_trained,
        "auc_standard_unlearned"  : auc_std_unlearned,
        "auc_forget_trained"      : auc_forget_trained,
        "auc_forget_unlearned"    : auc_forget_unlearned,
        "advantage_forget_trained"   : 2 * auc_forget_trained   - 1,
        "advantage_forget_unlearned" : 2 * auc_forget_unlearned - 1,
        "mean_scores" : mean,
    }

    print(f"\n── Metric: {metric.upper()} {'─'*60}")

    print(f"\n  [Standard MIA: retain_train vs test]")
    print(f"    AUC  trained   model : {auc_std_trained:.4f}")
    print(f"    AUC  unlearned model : {auc_std_unlearned:.4f}")
    print(f"    (Should stay similar — retain set is untouched)")

    print(f"\n  [Forget MIA: forget_train vs test]  ← KEY METRIC")
    print(f"    AUC  trained   model : {auc_forget_trained:.4f}  "
          f"(advantage {2*auc_forget_trained-1:.4f})")
    print(f"    AUC  unlearned model : {auc_forget_unlearned:.4f}  "
          f"(advantage {2*auc_forget_unlearned-1:.4f})")
    print(f"    Target: AUC ≈ 0.5 on unlearned model")

    print(f"\n  [Mean scores per population]")
    for pop in ("forget", "retain", "test"):
        print(f"    {pop:7s} → trained: {mean['trained'][pop]:.4f} | "
              f"unlearned: {mean['unlearned'][pop]:.4f}")

print(f"\n{'='*80}")
print("VERDICT GUIDE:")
print("  forget AUC on unlearned ≈ 0.5  →  forget samples indistinguishable from non-members ✓")
print("  forget AUC on unlearned >> 0.5 →  forget samples still look like members ✗")
print("  standard MIA AUC unchanged     →  retain performance preserved ✓")
print(f"{'='*80}\n")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE METRICS
# ──────────────────────────────────────────────────────────────────────────────
json_path = os.path.join(OUTPUT_DIR, "inference_attack_metrics.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"Metrics saved → {json_path}")

csv_path = os.path.join(OUTPUT_DIR, "inference_attack_metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Metric",
        "AUC_standard_trained", "AUC_standard_unlearned",
        "AUC_forget_trained",   "AUC_forget_unlearned",
        "Advantage_forget_trained", "Advantage_forget_unlearned",
        "Mean_forget_trained",  "Mean_forget_unlearned",
        "Mean_retain_trained",  "Mean_retain_unlearned",
        "Mean_test_trained",    "Mean_test_unlearned",
    ])
    for metric in METRICS:
        r = results[metric]
        m = r["mean_scores"]
        writer.writerow([
            metric,
            f"{r['auc_standard_trained']:.4f}",
            f"{r['auc_standard_unlearned']:.4f}",
            f"{r['auc_forget_trained']:.4f}",
            f"{r['auc_forget_unlearned']:.4f}",
            f"{r['advantage_forget_trained']:.4f}",
            f"{r['advantage_forget_unlearned']:.4f}",
            f"{m['trained']['forget']:.4f}",
            f"{m['unlearned']['forget']:.4f}",
            f"{m['trained']['retain']:.4f}",
            f"{m['unlearned']['retain']:.4f}",
            f"{m['trained']['test']:.4f}",
            f"{m['unlearned']['test']:.4f}",
        ])
print(f"CSV saved     → {csv_path}")

# ──────────────────────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(len(METRICS), 2, figsize=(14, 4 * len(METRICS)))
fig.suptitle("Membership Inference Attack — Score Distributions", fontsize=14, y=1.01)

colors = {"forget": "#e74c3c", "retain": "#2ecc71", "test": "#3498db"}

for row, metric in enumerate(METRICS):
    for col, (model_name, model_label) in enumerate(
        [("trained", "Trained Model"), ("unlearned", "Unlearned Model")]
    ):
        ax = axes[row, col]
        for pop in ("forget", "retain", "test"):
            s = scores[model_name][pop][metric]
            ax.hist(s, bins=40, alpha=0.55, color=colors[pop],
                    label=f"{pop} (n={len(s)})", density=True)

        auc_f = results[metric][f"auc_forget_{model_name}"]
        ax.set_title(f"{model_label} | {metric} | forget AUC={auc_f:.3f}", fontsize=10)
        ax.set_xlabel("Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

        # Annotate: on unlearned model, are forget and test distributions close?
        if model_name == "unlearned":
            f_mean = scores["unlearned"]["forget"][metric].mean()
            t_mean = scores["unlearned"]["test"][metric].mean()
            ax.axvline(f_mean, color=colors["forget"], linestyle="--", linewidth=1.2)
            ax.axvline(t_mean, color=colors["test"],   linestyle="--", linewidth=1.2)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "inference_attack_distributions.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"Plot saved    → {plot_path}")
plt.close()

# ── Summary bar chart: forget AUC before vs after unlearning ─────────────────
fig, ax = plt.subplots(figsize=(8, 4))
x      = np.arange(len(METRICS))
width  = 0.35

aucs_trained   = [results[m]["auc_forget_trained"]   for m in METRICS]
aucs_unlearned = [results[m]["auc_forget_unlearned"] for m in METRICS]

bars1 = ax.bar(x - width/2, aucs_trained,   width, label="Trained",   color="#e74c3c", alpha=0.8)
bars2 = ax.bar(x + width/2, aucs_unlearned, width, label="Unlearned", color="#3498db", alpha=0.8)

ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, label="Random (0.5)")
ax.set_xticks(x)
ax.set_xticklabels(METRICS)
ax.set_ylabel("Forget MIA AUC")
ax.set_title("Forget Set MIA AUC: Trained vs Unlearned\n"
             "(closer to 0.5 on unlearned = better forgetting)")
ax.set_ylim(0, 1.05)
ax.legend()

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)

plt.tight_layout()
summary_path = os.path.join(OUTPUT_DIR, "forget_mia_auc_summary.png")
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
print(f"Summary plot  → {summary_path}")
plt.close()