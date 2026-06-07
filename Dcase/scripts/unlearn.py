import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset import get_forget_splits, get_retain_splits
from src.model   import DcaseMultimodalModel
from src.labels  import NUM_CLASSES

DEVICE = torch.device(os.environ.get("DEVICE", "cuda") if torch.cuda.is_available() else "cpu")

BASE_DIR             = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINED_MODEL_PATH   = os.path.join(BASE_DIR, "models", "dcase_trained.pth")
UNLEARNED_MODEL_PATH = os.path.join(BASE_DIR, "models", "dcase_unlearned_naive.pth")

BATCH_SIZE = 16
LR         = 1e-5
EPOCHS     = int(os.environ.get("EPOCHS", 40))
PATIENCE   = 10
T          = 1.0           

C = NUM_CLASSES             

forget_train, forget_val, forget_test = get_forget_splits(seed=42)
retain_train, retain_val, retain_test = get_retain_splits(seed=42)

# Create loaders
forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)
retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4)

def masking(logits, forget_labels):
    mask = torch.zeros_like(logits)
    if logits.size(0) == forget_labels.size(0):
        mask[torch.arange(logits.size(0)), forget_labels] = float("-inf")
    else:
        for c in forget_labels.unique():
            mask[:, c] = float("-inf")
    return logits + mask

def MD_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
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
        
def UKR_loss(teacher_logits, student_logits, T, reduction="batchmean"):
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

model_ori = DcaseMultimodalModel(num_classes=C).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))

model_unlearn = deepcopy(model_ori)

for p in model_ori.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model_unlearn.parameters(), lr=LR)

def func_max(out, forget_labels):
    all_logits = torch.stack(
        [out["fusion_logits"], out["audio_logits"], out["video_logits"]], dim=1
    )
    B, _, C_dim = all_logits.shape
    all_probs = F.softmax(all_logits, dim=2)
    y = forget_labels.view(B, 1, 1)
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best_branch = probs_y.argmax(dim=1)
    chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]
    return chosen_logits

def func_max_retain(out, retain_labels):
    all_logits = torch.stack(
        [out["fusion_logits"], out["audio_logits"], out["video_logits"]], dim=1
    )
    B, _, C_dim = all_logits.shape
    all_probs = F.softmax(all_logits, dim=2)
    y = retain_labels.view(B, 1, 1)
    probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)
    best_branch = probs_y.argmax(dim=1)
    chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]
    return chosen_logits

def compute_unlearning_loss(model_ori, model_unlearn, batch_df, batch_dr):
    for k in batch_df:
        if isinstance(batch_df[k], torch.Tensor):
            batch_df[k] = batch_df[k].to(DEVICE)
    for k in batch_dr:
        if isinstance(batch_dr[k], torch.Tensor):
            batch_dr[k] = batch_dr[k].to(DEVICE)

    out_df_un = model_unlearn(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
    out_dr_un = model_unlearn(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)

    with torch.no_grad():
        out_df_ori = model_ori(batch_df["video"], batch_df["spectrogram"], return_intermediate=True)
        out_dr_ori = model_ori(batch_dr["video"], batch_dr["spectrogram"], return_intermediate=True)
        
    perm = torch.randperm(batch_dr["spectrogram"].size(0))
    rand_spec = batch_dr["spectrogram"][perm]

    with torch.no_grad():
        out_ori_random = model_ori(
            batch_dr["video"],
            rand_spec,
            return_intermediate=True
        )

    forget_labels = batch_df["label"]
    retain_labels = batch_dr["label"]

    loss_md_val = MD_loss(func_max(out_ori_random, forget_labels), out_df_un["fusion_logits"], T, forget_labels, reduction="batchmean")
    
    loss_multi_val = (
        MKR_loss(func_max_retain(out_dr_ori, retain_labels), out_dr_un["fusion_logits"], T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max_retain(out_dr_ori, retain_labels), out_dr_un["video_logits"], T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max_retain(out_dr_ori, retain_labels), out_dr_un["audio_logits"], T, forget_labels, reduction="batchmean")
    ) / 3.0
    
    loss_uni_val = 0.5 * (
        UKR_loss(func_max(out_df_ori, forget_labels), out_df_un["audio_logits"], T, reduction="batchmean") +
        UKR_loss(func_max(out_df_ori, forget_labels), out_df_un["video_logits"], T, reduction="batchmean")
    )

    total_loss = (loss_md_val) + (loss_multi_val) + (loss_uni_val)

    return {
        "train_loss": total_loss,
        "loss_md": loss_md_val,
        "loss_multi": loss_multi_val,
        "loss_uni": loss_uni_val
    }

best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

for epoch in range(EPOCHS):
    model_unlearn.train()

    running_train_loss = 0.0
    num_train_batches = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()
        out = compute_unlearning_loss(model_ori, model_unlearn, batch_df, batch_dr)
        out["train_loss"].backward()
        optimizer.step()
        
        running_train_loss += out["train_loss"].item()
        num_train_batches += 1
    
    avg_train_loss = running_train_loss / num_train_batches if num_train_batches > 0 else 0

    model_unlearn.eval()
    running_val_loss = 0.0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(model_ori, model_unlearn, batch_df, batch_dr)
            running_val_loss += out["train_loss"].item()
            num_val_batches += 1

    avg_val_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0

    print(f"Epoch {epoch} | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} |")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = deepcopy(model_unlearn.state_dict())
        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
        print(f"  --> Best val loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience {patience_counter}/{PATIENCE}")
        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

if best_model_state is not None:
    model_unlearn.load_state_dict(best_model_state)
    print("Restored best model execution weights.")

print("Unlearning complete.")
