# import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# from src.model import MultimodalFoodClassifier
# from src.dataset import ForgetDataset, RetainDataset
# from torch.utils.data import DataLoader
# from copy import deepcopy

# DEVICE=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# TRAINED_MODEL_PATH="/home/team2/Unlearning/newDir/models4/best_multimodal_food101_101cls.pth"

# UNLEARNED_MODEL_PATH="/home/team2/Unlearning/newDir/models4/unlearned_multimodal_neww_embed.pth"

# BATCH_SIZE = 16
# LR = 1e-5
# EPOCHS = 20

# forget_dataset = ForgetDataset()
# retain_dataset = RetainDataset()

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
#             batch_dr["image"],  # in logit_unlearn it's batch_dr["image"]
#             rand_ids,
#             rand_mask,
#             return_intermediate=True
#         )

#     #almost all same 

    
#     loss_md = mse(
#         out_ori_random["fused_emb"],
#         out_df_un["fused_emb"]
#     )

#     # ---- Multimodal retain (Dr) ----
#     loss_multi = mse(
#         out_dr_ori["fused_emb"],
#         out_dr_un["fused_emb"]
#     )

#     # ---- Unimodal retain (Df image) ----
#     loss_uni = mse(
#         out_df_ori["text_emb"],
#         out_df_un["text_emb"]
#     )

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




# import torch
# import torch.nn as nn
# import torch.nn.functional as F 
# from src.model import MultimodalFoodClassifier
# from src.dataset import ForgetDataset, RetainDataset
# from torch.utils.data import DataLoader
# from copy import deepcopy

# DEVICE=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# TRAINED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/models5/best_multimodal_food101_101cls.pth"
# UNLEARNED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/models5/unlearned_multimodal_embed.pth"

# BATCH_SIZE = 8
# LR = 1e-5
# EPOCHS = 20

# forget_dataset = ForgetDataset()
# retain_dataset = RetainDataset()

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

# for p in model_ori.parameters():
#     p.requires_grad=False

# for p in model_unlearn.classifier.parameters():
#     p.requires_grad = False


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

#     loss_md = mse(
#         out_ori_random["fused_emb"],
#         out_df_un["fused_emb"]
#     )

#     # ---- Multimodal retain (Dr) ----
#     loss_multi = mse(
#         out_dr_ori["fused_emb"],
#         out_dr_un["fused_emb"]
#     )

#     # ---- Unimodal retain (Df image) ----
#     # loss_uni = mse(
#     #     out_df_ori["text_emb"],
#     #     out_df_un["text_emb"]
#     # )

#     loss_uni = 0

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
#         # f"Uni {out['loss_uni'].item():.4f}"
#     )

# torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
# print("Unlearning complete. Model saved.")

from copy import deepcopy
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.model_new_r import MultimodalFoodClassifier
from src.dataset import ForgetDataset, RetainDataset

# --------------------
# CONFIG
# --------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

_t_start = time.perf_counter()

TRAINED_MODEL_PATH = "/home/team2/Unlearning/Food101-files/models/best_multimodal_food101.pth"
UNLEARNED_MODEL_PATH = "/home/team2/Unlearning/Food101-files/models/unlearned_multimodal_embed.pth"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 40
PATIENCE = 5

# --------------------
# DATA
# --------------------
forget_dataset = ForgetDataset()
retain_dataset = RetainDataset()

def split_dataset(dataset, val_ratio=0.2):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

forget_train, forget_val = split_dataset(forget_dataset)
retain_train, retain_val = split_dataset(retain_dataset)

forget_train_loader = DataLoader(forget_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
forget_val_loader   = DataLoader(forget_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

retain_train_loader = DataLoader(retain_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
retain_val_loader   = DataLoader(retain_val,   batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

# --------------------
# MODELS
# --------------------
model_ori = MultimodalFoodClassifier(num_classes=101).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
model_ori.eval()

model_unlearn = deepcopy(model_ori)

for p in model_ori.parameters():
    p.requires_grad = False

optimizer = torch.optim.Adam(model_unlearn.parameters(), lr=LR)
mse = nn.MSELoss()

# --------------------
# LOSS FUNCTION
# --------------------
def compute_unlearning_loss(model_ori, model_unlearn, batch_df, batch_dr):
    for k in batch_df:
        batch_df[k] = batch_df[k].to(DEVICE)
    for k in batch_dr:
        batch_dr[k] = batch_dr[k].to(DEVICE)

    # ---- Unlearn outputs ----
    out_df_un = model_unlearn(
        batch_df["image"],
        batch_df["input_ids"],
        batch_df["attention_mask"],
        return_intermediate=True
    )

    out_dr_un = model_unlearn(
        batch_dr["image"],
        batch_dr["input_ids"],
        batch_dr["attention_mask"],
        return_intermediate=True
    )

    # ---- Original outputs ----
    with torch.no_grad():
        out_df_ori = model_ori(
            batch_df["image"],
            batch_df["input_ids"],
            batch_df["attention_mask"],
            return_intermediate=True
        )

        out_dr_ori = model_ori(
            batch_dr["image"],
            batch_dr["input_ids"],
            batch_dr["attention_mask"],
            return_intermediate=True
        )

    # ---- Random pairing ----
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

    # ---- Losses ----
    loss_md = mse(
        out_ori_random["fused_emb"],
        out_df_un["fused_emb"]
    )

    loss_multi = mse(
        out_dr_ori["fused_emb"],
        out_dr_un["fused_emb"]
    )

    loss_uni = 0.5 * (
        mse(out_df_ori["text_emb"],  out_df_un["text_emb"]) +
        mse(out_df_ori["image_emb"], out_df_un["image_emb"])
    )

    total_loss = loss_md + loss_multi + loss_uni

    return {
        "train_loss": total_loss,
        "loss_md": loss_md,
        "loss_multi": loss_multi,
        "loss_uni": loss_uni
    }

# --------------------
# EARLY STOPPING SETUP
# --------------------
best_val_loss = float('inf')
patience_counter = 0
best_model_state = None

# --------------------
# TRAIN LOOP
# --------------------
for epoch in range(EPOCHS):

    # ===== TRAIN =====
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

    avg_train_loss = (
        running_train_loss / num_train_batches
        if num_train_batches > 0 else 0
    )

    # ===== VALIDATION =====
    model_unlearn.eval()
    running_val_loss = 0.0
    num_val_batches = 0

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

    avg_val_loss = (
        running_val_loss / num_val_batches
        if num_val_batches > 0 else 0
    )

    print(
        f"Epoch {epoch} | "
        f"Train Loss {avg_train_loss:.4f} | "
        f"Val Loss {avg_val_loss:.4f}"
    )

    # ===== EARLY STOPPING =====
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        best_model_state = deepcopy(model_unlearn.state_dict())

        torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
        print("  --> Best val loss improved. Model saved.")
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# Restore best weights
if best_model_state is not None:
    model_unlearn.load_state_dict(best_model_state)
    print("Restored best model weights.")

print("Unlearning complete.")


_t_end = time.perf_counter()
_elapsed = _t_end - _t_start
_h, _rem = divmod(int(_elapsed), 3600)
_m, _s   = divmod(_rem, 60)
print(f"\n{'='*60}")
print(f"  Runtime (train.py): {_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)")
print(f"{'='*60}")