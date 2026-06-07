import torch
import torch.nn as nn
import torch.nn.functional as F 
import time
from src.model_new_r import MultimodalFoodClassifier
from src.dataset import ForgetDataset, RetainDataset
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_PATH="/home/team2/Unlearning/Food101-files/models/model_trained.pth"
UNLEARNED_MODEL_PATH="/home/team2/Unlearning/Food101-files/models/unlearned_4T.pth"

_t_start = time.perf_counter()

BATCH_SIZE = 16
LR = 5*1e-5
EPOCHS = 40
PATIENCE = 10

forget_dataset = ForgetDataset()
retain_dataset = RetainDataset()

# Split datasets 80:20
def split_dataset(dataset, val_ratio=0.2):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

forget_train, forget_val = split_dataset(forget_dataset)
retain_train, retain_val = split_dataset(retain_dataset)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

class WeightNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output raw score
        )

    def forward(self, x):
        return self.fc(x)

def masking(logits, forget_labels):
    """
    logits: (N, K)
    forget_labels: (N,)
    """
    mask = torch.zeros_like(logits)
    mask[torch.arange(logits.size(0)), forget_labels] = float("-inf")
    return logits + mask

def MD_loss(student_logits, T, forget_labels, reduction="batchmean"):
    """
    MD: retain mismatched distillation
    """
    teacher_logits = torch.ones_like(student_logits).to(DEVICE)
    student_log_prob = F.log_softmax(student_logits / T, dim=1)

    masked_teacher_logits = masking(teacher_logits, forget_labels)
    teacher_prob = F.softmax(masked_teacher_logits / T, dim=1)
    
    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    
    if reduction == "none":
        return loss.sum(dim=1) # Sum over classes to get per-sample loss
    elif reduction == "batchmean":
        return loss.sum() / student_logits.shape[0] # batchmean is sum over batch / batch_size
    else:
        return loss.mean()
        

def UKR_loss(teacher_logits, student_logits, T, reduction="batchmean"):
    """
    UKR: forget data unlearning
    """
    student_log_prob = F.log_softmax(student_logits / T, dim=1)
    teacher_prob = F.softmax(teacher_logits / T, dim=1)

    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    
    if reduction == "none":
        return loss.sum(dim=1)
    elif reduction == "batchmean":
        return loss.sum() / student_logits.shape[0]
    else:
        return loss.mean()


def MKR_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    """
    MKR: retain data preservation
    """
    student_log_prob = F.log_softmax(student_logits / T, dim=1)

    masked_teacher_logits = masking(teacher_logits, forget_labels)
    teacher_prob = F.softmax(masked_teacher_logits / T, dim=1)

    loss = F.kl_div(student_log_prob, teacher_prob, reduction="none") * (T * T)
    
    if reduction == "none":
        return loss.sum(dim=1)
    elif reduction == "batchmean":
        return loss.sum() / student_logits.shape[0]
    else:
        return loss.mean()



model_ori=MultimodalFoodClassifier(num_classes=101).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
# model_ori.eval()

model_unlearn=deepcopy(model_ori)
#model_uni=deepcopy(model_ori)

for p in model_ori.parameters():
    p.requires_grad=False

# Initialize Weight Networks
# MD (a1): 3 vectors * 20 classes = 60
net_a1 = WeightNet(input_dim=3*101).to(DEVICE)
# MKR (a2): 3 branches * 3 vectors * 20 classes = 180
net_a2 = WeightNet(input_dim=6*101).to(DEVICE)
# UKR (a3): 2 branches * 3 vectors * 20 classes = 120
net_a3 = WeightNet(input_dim=6*101).to(DEVICE)

# Add parameters to optimizer
optimizer = torch.optim.Adam(
    list(model_unlearn.parameters()) + 
    list(net_a1.parameters()) + 
    list(net_a2.parameters()) + 
    list(net_a3.parameters()), 
    lr=LR
)

def func_max(out, forget_labels):
    """
    out: dict containing fusion_logits/text_logits/image_logits, each [B, C]
    forget_labels: [B] ground-truth class indices (forget class per sample)

    returns: chosen_logits [B, C]
    """

    # stack logits: [B, 3, C]
    all_logits = torch.stack(
        [out["fusion_logits"], out["text_logits"], out["image_logits"]],
        dim=1
    )

    B, _, C = all_logits.shape

    # softmax over classes: [B, 3, C]
    all_probs = F.softmax(all_logits, dim=2)

    # gather prob of the forget label for each branch
    # forget_labels: [B] -> [B, 1, 1] for broadcasting
    y = forget_labels.view(B, 1, 1)

    # probs_y: [B, 3]
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)

    # pick branch with highest prob for forget class
    best_branch = probs_y.argmax(dim=1)  # [B]

    # select logits from best branch: [B, C]
    chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]

    return chosen_logits

def func_max_teacher_logits(out, target_labels):
    """
    out: dict with logits
    target_labels: [B]
    Selects branch with highest LOGIT value for the target label.
    """
    # stack logits: [B, 3, C]
    all_logits = torch.stack(
        [out["fusion_logits"], out["text_logits"], out["image_logits"]],
        dim=1
    ) # [B, 3, C]

    B, _, C = all_logits.shape
    # indices for gather: [B, 3, 1]
    y = target_labels.view(B, 1, 1).expand(B, 3, 1)

    # Gather logits for the target class: [B, 3] (squeezing dim 2)
    logits_y = all_logits.gather(dim=2, index=y).squeeze(2)

    # branch w/ max logit
    best_branch = logits_y.argmax(dim=1) # [B]

    # select [B, C]
    chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]

    return chosen_logits


# --- Dynamic Weighting Logic ---

def get_diff_vectors(student_logits, teacher_logits, gt_one_hot):
    """
    Returns concatenated difference vectors:
    1. GT - Student (Prob)
    2. GT - Teacher (Prob)
    3. Student - Teacher (Prob)
    """
    student_prob = F.softmax(student_logits, dim=1)
    teacher_prob = F.softmax(teacher_logits, dim=1)
    
    diff1 = gt_one_hot - student_prob
    diff2 = gt_one_hot - teacher_prob
    diff3 = student_prob - teacher_prob
    
    return torch.cat([diff1, diff2, diff3], dim=1) # [B, 60]


def compute_unlearning_loss(model_ori, model_unlearn, net_a1, net_a2, net_a3, batch_df, batch_dr):
    """
    Calculates the total unlearning loss and its components.
    Used for both training (with grad) and validation (no grad).
    """
    mse = nn.MSELoss()

    for k in batch_df:
        batch_df[k] = batch_df[k].to(DEVICE)
    for k in batch_dr:
        batch_dr[k] = batch_dr[k].to(DEVICE)

    out_df_un=model_unlearn(batch_df["image"], batch_df["input_ids"], batch_df["attention_mask"], return_intermediate=True)
    out_dr_un=model_unlearn(batch_dr["image"], batch_dr["input_ids"], batch_dr["attention_mask"], return_intermediate=True)

    with torch.no_grad():
        out_df_ori=model_ori(batch_df["image"], batch_df["input_ids"], batch_df["attention_mask"], return_intermediate=True)
        out_dr_ori=model_ori(batch_dr["image"], batch_dr["input_ids"], batch_dr["attention_mask"], return_intermediate=True)
        
        # Get one-hot ground truth for difference vectors
        num_classes = 101
        gt_one_hot = F.one_hot(batch_df["label"], num_classes=num_classes).float()
        
    perm = torch.randperm(batch_dr["input_ids"].size(0))
    rand_ids = batch_dr["input_ids"][perm]
    rand_mask = batch_dr["attention_mask"][perm]

    with torch.no_grad():
        out_ori_random = model_ori(
            batch_dr["image"],
            rand_ids,
            rand_mask,
            return_intermediate=True
        )

    T = 4.0
    forget_labels = batch_df["label"]

    # Calculate difference vectors for each branch
    
    # MD Inputs for Difference Vector:
    # Student: out_df_un["fusion_logits"]
    # Teacher: func_max(out_ori_random, forget_labels) -> this is the target used in loss.
    # GT: gt_one_hot
    md_teacher_logits = torch.ones_like(out_df_un["fusion_logits"])
    md_teacher_logits = masking(md_teacher_logits,forget_labels)
    md_teacher_logits = F.softmax(md_teacher_logits,dim=1)

    vec_md = get_diff_vectors(
        out_df_un["fusion_logits"], 
        md_teacher_logits, 
        gt_one_hot
    ) # [B, 60]
    
    # MKR Inputs for Difference Vector:
    # MKR uses Retain Set (batch_dr).
    gt_one_hot_retain = F.one_hot(batch_dr["label"], num_classes=101).float()
    
    vec_mkr_fusion = get_diff_vectors(out_dr_un["fusion_logits"], func_max_teacher_logits(out_dr_ori, batch_dr["label"]), gt_one_hot_retain)
    vec_mkr_text = get_diff_vectors(out_dr_un["text_logits"], func_max_teacher_logits(out_dr_ori, batch_dr["label"]), gt_one_hot_retain)
    vec_mkr_image = get_diff_vectors(out_dr_un["image_logits"], func_max_teacher_logits(out_dr_ori, batch_dr["label"]), gt_one_hot_retain)
    
    vec_mkr_uni = torch.cat([vec_mkr_text, vec_mkr_image], dim=1) # [B, 180]
    vec_mkr = vec_mkr_fusion
    # UKR Inputs for Difference Vector:
    # UKR uses Forget Set (batch_df) but preserves Uni-modal performance.
    vec_ukr_text = get_diff_vectors(out_df_un["text_logits"], func_max(out_df_ori, forget_labels), gt_one_hot)
    vec_ukr_image = get_diff_vectors(out_df_un["image_logits"], func_max(out_df_ori, forget_labels), gt_one_hot)
    vec_ukr = torch.cat([vec_ukr_text, vec_ukr_image], dim=1) # [B, 120]

    # Predict weights
    raw_a1 = net_a1(vec_mkr).mean()
    raw_a2 = net_a2(vec_ukr).mean()
    raw_a3 = net_a3(vec_mkr_uni).mean()
    
    scores = torch.stack([raw_a1, raw_a2, raw_a3]) # [3]
    weights = F.softmax(scores, dim=0) * 3
    a1, a2, a3 = weights[0], weights[1], weights[2]

    # Calculate Losses (reduction='batchmean' effectively, but weighted)
    loss_md_val = MD_loss(out_df_un["fusion_logits"], T, forget_labels, reduction="batchmean")
    
    loss_multi_val = (
        MKR_loss(func_max_teacher_logits(out_dr_ori, batch_dr["label"]), out_dr_un["fusion_logits"], T, forget_labels, reduction="batchmean"))
    
    loss_uni_val = 0.5 * (
        UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["text_logits"], T, reduction="batchmean") +
        UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["image_logits"], T, reduction="batchmean")
    )

    loss_mkr_uni_val = (MKR_loss(func_max_teacher_logits(out_dr_ori, batch_dr["label"]), out_dr_un["image_logits"], T, forget_labels, reduction="batchmean") +
                        MKR_loss(func_max_teacher_logits(out_dr_ori, batch_dr["label"]), out_dr_un["text_logits"], T, forget_labels, reduction="batchmean")) / 2.0

    loss_uni_ce = (
        torch.nn.functional.cross_entropy(out_dr_un["image_logits"], batch_dr["label"]) +
        torch.nn.functional.cross_entropy(out_dr_un["text_logits"], batch_dr["label"])
    )
    total_loss = (loss_md_val) + (a2 * loss_uni_val) + (a1 * loss_multi_val) + a3*loss_mkr_uni_val+loss_uni_ce

    return {
        "train_loss": total_loss,
        "loss_md": loss_md_val,
        "loss_multi": loss_multi_val,
        "loss_uni": loss_uni_val,
        "a1": a1, # Keep as tensors if needed for grad, but here primarily for value
        "a2": a2,
        "a3": a3
    }


# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
best_nets_state = {}

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model_unlearn.train()
    net_a1.train()
    net_a2.train()
    net_a3.train()
    
    running_train_loss = 0.0
    num_train_batches = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori,
            model_unlearn,
            net_a1, net_a2, net_a3,
            batch_df,
            batch_dr
        )

        out["train_loss"].backward()
        optimizer.step()
        
        running_train_loss += out["train_loss"].item()
        num_train_batches += 1
    
    avg_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else 0

    # --- Validation Phase ---
    model_unlearn.eval()
    net_a1.eval()
    net_a2.eval()
    net_a3.eval()
    
    running_val_loss = 0.0
    num_val_batches = 0
    
    # For reporting last batch metrics (optional)
    last_val_out = None

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(
                model_ori,
                model_unlearn,
                net_a1, net_a2, net_a3,
                batch_df,
                batch_dr
            )
            running_val_loss += out["train_loss"].item()
            num_val_batches += 1
            last_val_out = out

    avg_val_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0

    # Print status
    # We use the a1, a2, a3 from the last validation batch as a sample reference
    a1_val = last_val_out['a1'].item() if last_val_out else 0
    a2_val = last_val_out['a2'].item() if last_val_out else 0
    a3_val = last_val_out['a3'].item() if last_val_out else 0
    
    print(
        f"Epoch {epoch} | "
        f"Train Loss {avg_train_loss:.4f} | "
        f"Val Loss {avg_val_loss:.4f} | "
        # f"Multi {last_val_out['loss_multi'].item():.4f} | "
        # f"Uni {last_val_out['loss_uni'].item():.4f} | "
        f"a1 {a1_val:.2f} | a2 {a2_val:.2f} | a3 {a3_val:.2f}"
    )

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = deepcopy(model_unlearn.state_dict())
        # Save nets if needed (optional, but good for resuming)
        best_nets_state = {
            'a1': deepcopy(net_a1.state_dict()),
            'a2': deepcopy(net_a2.state_dict()),
            'a3': deepcopy(net_a3.state_dict())
        }
        # Save best model to disk immediately
        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
        print(f"  --> Best val loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Restore best model
if best_model_state is not None:
    model_unlearn.load_state_dict(best_model_state)
    print("Restored best model execution weights.")

print("Unlearning complete.")

_t_end = time.perf_counter()
_elapsed = _t_end - _t_start
_h, _rem = divmod(int(_elapsed), 3600)
_m, _s   = divmod(_rem, 60)
print(f"\n{'='*60}")
print(f"  Runtime (train.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
print(f"{'='*60}")