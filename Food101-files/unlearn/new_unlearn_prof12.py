# import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# from src.model_new_r import MultimodalFoodClassifier
# from src.dataset import ForgetDataset, RetainDataset
# from torch.utils.data import DataLoader
# from copy import deepcopy

# DEVICE=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# TRAINED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/newArch/models1/best_multimodal_food101_20cls.pth"
# UNLEARNED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/newArch/models_prof1/unlearned_multimodal_logits_mod2.pth"

# BATCH_SIZE = 16
# LR = 1e-4
# EPOCHS = 20

# forget_dataset = ForgetDataset()
# retain_dataset = RetainDataset()

# def masking(logits, forget_labels):
#     """
#     logits: (N, K)
#     forget_labels: (N,)
#     """
#     mask = torch.zeros_like(logits)
#     mask[torch.arange(logits.size(0)), forget_labels] = float("-inf")
#     return logits + mask

# def MD_loss(teacher_logits, student_logits, T, forget_labels):
#     """
#     MD: retain mismatched distillation
#     """
#     student_log_prob = F.log_softmax(student_logits / T, dim=1)

#     masked_teacher_logits = masking(teacher_logits, forget_labels)
#     teacher_prob = F.softmax(masked_teacher_logits / T, dim=1)
#     N, K = student_logits.shape
#     # teacher_prob = torch.full(
#     #     size=(N, K),
#     #     fill_value=1.0 / K,
#     #     device=student_logits.device
#     # )

#     return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)


# def UKR_loss(teacher_logits, student_logits, T):
#     """
#     UKR: forget data unlearning
#     """
#     student_log_prob = F.log_softmax(student_logits / T, dim=1)
#     teacher_prob = F.softmax(teacher_logits / T, dim=1)

#     return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)


# def MKR_loss(teacher_logits, student_logits, T, forget_labels):
#     """
#     MKR: retain data preservation
#     """
#     student_log_prob = F.log_softmax(student_logits / T, dim=1)

#     masked_teacher_logits = masking(teacher_logits, forget_labels)
#     teacher_prob = F.softmax(masked_teacher_logits / T, dim=1)

#     return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)



# forget_loader = DataLoader(
#     forget_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True
# )
# retain_loader = DataLoader(
#     retain_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=True,
#     drop_last=True
# )

# model_ori=MultimodalFoodClassifier(num_classes=20).to(DEVICE)
# model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
# # model_ori.eval()

# model_unlearn=deepcopy(model_ori)
# #model_uni=deepcopy(model_ori)

# for p in model_ori.parameters():
#     p.requires_grad=False

# optimizer = torch.optim.Adam(model_unlearn.parameters(), lr=LR)


# def unlearning_train_step(model_ori, model_unlearn, batch_df, batch_dr):
#     mse = nn.MSELoss()

#     for k in batch_df:
#         batch_df[k] = batch_df[k].to(DEVICE)
#     for k in batch_dr:
#         batch_dr[k] = batch_dr[k].to(DEVICE)

#     out_df_un=model_unlearn(batch_df["image"], batch_df["input_ids"], batch_df["attention_mask"], return_intermediate=True)
#     out_dr_un=model_unlearn(batch_dr["image"], batch_dr["input_ids"], batch_dr["attention_mask"], return_intermediate=True)

#     with torch.no_grad():
#         out_df_ori=model_ori(batch_df["image"], batch_df["input_ids"], batch_df["attention_mask"], return_intermediate=True)
#         out_dr_ori=model_ori(batch_dr["image"], batch_dr["input_ids"], batch_dr["attention_mask"], return_intermediate=True)

#     perm = torch.randperm(batch_dr["input_ids"].size(0))
#     rand_ids = batch_dr["input_ids"][perm]
#     rand_mask = batch_dr["attention_mask"][perm]

#     with torch.no_grad():
#         out_ori_random = model_ori(
#             batch_dr["image"],
#             rand_ids,
#             rand_mask,
#             return_intermediate=True
#         )

#     T = 4.0
#     forget_labels = batch_df["label"]



#     def func_max(out, forget_labels):
#         """
#         out: dict containing fusion_logits/text_logits/image_logits, each [B, C]
#         forget_labels: [B] ground-truth class indices (forget class per sample)

#         returns: chosen_logits [B, C]
#         """

#         # stack logits: [B, 3, C]
#         all_logits = torch.stack(
#             [out["fusion_logits"], out["text_logits"], out["image_logits"]],
#             dim=1
#         )

#         B, _, C = all_logits.shape

#         # softmax over classes: [B, 3, C]
#         all_probs = F.softmax(all_logits, dim=2)

#         # gather prob of the forget label for each branch
#         # forget_labels: [B] -> [B, 1, 1] for broadcasting
#         y = forget_labels.view(B, 1, 1)

#         # probs_y: [B, 3]
#         probs_y = all_probs.gather(dim=2, index=y.expand(B, 3, 1)).squeeze(2)

#         # pick branch with highest prob for forget class
#         best_branch = probs_y.argmax(dim=1)  # [B]

#         # select logits from best branch: [B, C]
#         chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]

#         return chosen_logits

#     def func_max2(out):
#         # confidence per sample
#         fusion_conf = F.softmax(out["fusion_logits"], dim=1).max(dim=1).values  # [B]
#         text_conf   = F.softmax(out["text_logits"], dim=1).max(dim=1).values    # [B]
#         image_conf  = F.softmax(out["image_logits"], dim=1).max(dim=1).values   # [B]

#         # pick best branch per sample
#         confs = torch.stack([fusion_conf, text_conf, image_conf], dim=1)  # [B, 3]
#         best_branch = confs.argmax(dim=1)  # [B]

#         # stack logits: [B, 3, C]
#         all_logits = torch.stack(
#             [out["fusion_logits"], out["text_logits"], out["image_logits"]],
#             dim=1
#         )

#         # select [B, C]
#         B = all_logits.size(0)
#         chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]

#         return chosen_logits




#     loss_md = MD_loss(out_ori_random["fusion_logits"], out_df_un["fusion_logits"], T, forget_labels)
#     #loss_multi = (MKR_loss(func_max(out_dr_ori,forget_labels), out_dr_un["fusion_logits"], T, forget_labels)+MKR_loss(func_max(out_dr_ori,forget_labels), out_dr_un["text_logits"], T, forget_labels)+MKR_loss(func_max(out_dr_ori,forget_labels), out_dr_un["image_logits"], T, forget_labels))/3.0
#     loss_multi = (MKR_loss(func_max2(out_dr_ori), out_dr_un["fusion_logits"], T, forget_labels)+MKR_loss(func_max2(out_dr_ori), out_dr_un["image_logits"], T, forget_labels)+MKR_loss(func_max2(out_dr_ori), out_dr_un["text_logits"], T, forget_labels))/3.0
#     loss_uni = 0.5*(UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["text_logits"], T)+UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["image_logits"], T))

#     total_loss = loss_md + loss_multi + loss_uni

#     return {
#         "train_loss": total_loss,
#         "loss_md": loss_md,
#         "loss_multi": loss_multi,
#         "loss_uni": loss_uni
#     }



# for epoch in range(EPOCHS):
#     for batch_df, batch_dr in zip(forget_loader, retain_loader):

#         optimizer.zero_grad()

#         out = unlearning_train_step(
#             model_ori,
#             model_unlearn,
#             batch_df,
#             batch_dr
#         )

#         out["train_loss"].backward()
#         optimizer.step()

#     print(
#         f"Epoch {epoch} | "
#         f"Total {out['train_loss'].item():.4f} | "
#         f"MD {out['loss_md'].item():.4f} | "
#         f"Multi {out['loss_multi'].item():.4f} | "
#         f"Uni {out['loss_uni'].item():.4f}"
#     )

# torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
# print("Unlearning complete. Model saved.")





import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.model_new_r import MultimodalFoodClassifier
from src.dataset import ForgetDataset, RetainDataset
from torch.utils.data import DataLoader, random_split
from copy import deepcopy

DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_PATH="/home/team2/Unlearning/Food101-files/models/model_trained.pth"
UNLEARNED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/full_dataset/models/unlearned_multimodal_11.pth"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 40
PATIENCE = 5
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


def masking(logits, forget_labels):
    """
    logits: (N, K)
    forget_labels: (N,)
    """
    mask = torch.zeros_like(logits)
    mask[torch.arange(logits.size(0)), forget_labels] = float("-inf")
    return logits + mask

def MD_loss(teacher_logits, student_logits, T, forget_labels, reduction="batchmean"):
    """
    MD: retain mismatched distillation
    """
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


# Add parameters to optimizer
optimizer = torch.optim.Adam(model_unlearn.parameters(), lr=LR)

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

def func_max2(out,retain_labels):
    # confidence per sample
    fusion_conf = F.softmax(out["fusion_logits"], dim=1).max(dim=1).values  # [B]
    text_conf   = F.softmax(out["text_logits"], dim=1).max(dim=1).values    # [B]
    image_conf  = F.softmax(out["image_logits"], dim=1).max(dim=1).values   # [B]

    # pick best branch per sample
    confs = torch.stack([fusion_conf, text_conf, image_conf], dim=1)  # [B, 3]
    best_branch = confs.argmax(dim=1)  # [B]

    # stack logits: [B, 3, C]
    all_logits = torch.stack(
        [out["fusion_logits"], out["text_logits"], out["image_logits"]],
        dim=1
    )

    # select [B, C]
    B = all_logits.size(0)
    chosen_logits = all_logits[torch.arange(B, device=all_logits.device), best_branch]

    return chosen_logits

def func_max_retain(out, retain_labels):
    """
    out: dict containing fusion_logits/text_logits/image_logits, each [B, C]
    retain_labels: [B] ground-truth retain class indices

    returns: chosen_logits [B, C]
    """

    # stack logits: [B, 3, C]
    all_logits = torch.stack(
        [out["fusion_logits"], out["text_logits"], out["image_logits"]],
        dim=1
    )

    B, _, C = all_logits.shape

    # softmax over classes
    all_probs = F.softmax(all_logits, dim=2)  # [B, 3, C]

    # reshape labels for gather
    y = retain_labels.view(B, 1, 1)

    # probability of retain label per branch → [B, 3]
    probs_y = all_probs.gather(
        dim=2,
        index=y.expand(B, 3, 1)
    ).squeeze(2)

    # select branch with highest retain-label probability
    best_branch = probs_y.argmax(dim=1)  # [B]

    # select logits from chosen branch
    chosen_logits = all_logits[
        torch.arange(B, device=all_logits.device),
        best_branch
    ]

    return chosen_logits

def compute_unlearning_loss(model_ori, model_unlearn, batch_df, batch_dr):
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

    T = 50.0
    forget_labels = batch_df["label"]
    retain_labels = batch_dr["label"]

    num_classes = 101
    B = forget_labels.size(0)
    device = forget_labels.device

    # Start with uniform over all classes
    forget_one_hot_mask = torch.full(
        (B, num_classes),
        fill_value=1.0 / (num_classes - 1),
        device=device
    )

    # Set forget class to 0
    forget_one_hot_mask[torch.arange(B), forget_labels] = 0.0
    def gt_one_hot_retain(retain_labels):
        retain_one_hot = F.one_hot(retain_labels, num_classes=101).float()
        return retain_one_hot

    retain_one_hot = gt_one_hot_retain(batch_dr["label"])
    forget_one_hot = gt_one_hot_retain(batch_df["label"])

    def kl(student_logits, teacher_prob, T):
        """
        MKR: retain data preservation
        """
        student_log_prob = F.log_softmax(student_logits / T, dim=1)
        #teacher_prob = F.softmax(teacher_logits / T, dim=1)

        return F.kl_div(student_log_prob, teacher_prob, reduction="batchmean") * (T * T)

    loss_md_val = kl(out_df_un["fusion_logits"], forget_one_hot_mask, T)
    #loss_md_val = MD_loss(func_max(out_ori_random,forget_labels), out_df_un["fusion_logits"], T, forget_labels, reduction="batchmean")
    
    loss_multi_val = (
        MKR_loss(func_max_retain(out_dr_ori,retain_labels), out_dr_un["fusion_logits"], T, forget_labels, reduction="batchmean") +
        3*MKR_loss(func_max_retain(out_dr_ori,retain_labels), out_dr_un["image_logits"], T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max_retain(out_dr_ori,retain_labels), out_dr_un["text_logits"], T, forget_labels, reduction="batchmean")
    )/(5.0)
    #loss_multi_val = (kl(out_dr_un["fusion_logits"], retain_one_hot, T)+kl(out_dr_un["image_logits"], retain_one_hot, T)+kl(out_dr_un["text_logits"], retain_one_hot, T))/3.0

    
    loss_uni_val = 0.5 * (
        UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["text_logits"], T, reduction="batchmean") +
        UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["image_logits"], T, reduction="batchmean")
    )
    #loss_uni_val = 0.5*(kl(out_df_un["text_logits"], forget_one_hot, T)+kl(out_df_un["image_logits"], forget_one_hot, T))


    total_loss = (loss_md_val) + (loss_multi_val) + (loss_uni_val)

    return {
        "train_loss": total_loss,
        "loss_md": loss_md_val,
        "loss_multi": loss_multi_val,
        "loss_uni": loss_uni_val
    }


# Early stopping variables
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None
best_nets_state = {}

for epoch in range(EPOCHS):
    # --- Training Phase ---
    model_unlearn.train()

    running_train_loss = 0.0
    num_train_batches = 0

    for batch_df, batch_dr in zip(forget_train_loader, retain_train_loader):
        optimizer.zero_grad()

        out = compute_unlearning_loss(
            model_ori,
            model_unlearn,
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
    
    running_val_loss = 0.0
    num_val_batches = 0
    
    # For reporting last batch metrics (optional)
    last_val_out = None

    with torch.no_grad():
        for batch_df, batch_dr in zip(forget_val_loader, retain_val_loader):
            out = compute_unlearning_loss(
                model_ori,
                model_unlearn,
                batch_df,
                batch_dr
            )
            running_val_loss += out["train_loss"].item()
            num_val_batches += 1
            last_val_out = out

    avg_val_loss = running_val_loss / num_val_batches if num_val_batches > 0 else 0

    
    print(
        f"Epoch {epoch} | "
        f"Train Loss {avg_train_loss:.4f} | "
        f"Val Loss {avg_val_loss:.4f} | "
    )

    # --- Early Stopping Check ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = deepcopy(model_unlearn.state_dict())
        # Save nets if needed (optional, but good for resuming)

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