import torch
import torch.nn as nn
import torch.nn.functional as F 
from src.model_new_r import MultimodalFoodClassifier
from src.dataset import ForgetDataset, RetainDataset
from torch.utils.data import DataLoader
from copy import deepcopy

DEVICE=torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
TRAINED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/newArch/models1/best_multimodal_food101_20cls.pth"
UNLEARNED_MODEL_PATH="/home/team2/Unlearning/newDirauth2/newArch/models_dynamic_alpha/unlearned_multimodal_logits_dynamic_alpha_trial2.pth"

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 20

forget_dataset = ForgetDataset()
retain_dataset = RetainDataset()

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



forget_loader = DataLoader(
    forget_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)
retain_loader = DataLoader(
    retain_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True
)

model_ori=MultimodalFoodClassifier(num_classes=20).to(DEVICE)
model_ori.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE))
# model_ori.eval()

model_unlearn=deepcopy(model_ori)
#model_uni=deepcopy(model_ori)

for p in model_ori.parameters():
    p.requires_grad=False

# Initialize Weight Networks
# MD (a1): 3 vectors * 20 classes = 60
net_a1 = WeightNet(input_dim=60).to(DEVICE)
# MKR (a2): 3 branches * 3 vectors * 20 classes = 180
net_a2 = WeightNet(input_dim=180).to(DEVICE)
# UKR (a3): 2 branches * 3 vectors * 20 classes = 120
net_a3 = WeightNet(input_dim=120).to(DEVICE)

# Add parameters to optimizer
optimizer = torch.optim.Adam(
    list(model_unlearn.parameters()) + 
    list(net_a1.parameters()) + 
    list(net_a2.parameters()) + 
    list(net_a3.parameters()), 
    lr=LR
)


def unlearning_train_step(model_ori, model_unlearn, net_a1, net_a2, net_a3, batch_df, batch_dr):
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
        # batch_df["label"] is [B]
        # We need [B, 20] one-hot
        num_classes = 20
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

    def func_max2(out):
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

    # Calculate difference vectors for each branch
    # For MD (a1): Fusion branch only. 
    # Use out_df_un (student) vs out_df_ori (teacher) on forget data? 
    # The loss uses out_ori_random vs out_df_un.
    # User said: "difference between ground truth (1 hot encoded value from the fusion branch of teacher logits) and student logits."
    # Wait, "1 hot encoded value from the fusion branch of teacher logits" logic is slightly ambiguous.
    # "ground truth (1 hot encoded value from the fusion branch of teacher logits)" might mean the teacher's prediction?
    # BUT "difference between ground truth AND teacher logits" implies GT is separate.
    # I will assume "Ground Truth" is the actual label y. 
    # The "MD" loss is computed on Forget Set. So we use `batch_df` inputs.
    # Note: MD loss uses `out_ori_random` as "teacher" (random pair). 
    # But usually "Teacher" implies the original model on the same input.
    # Let's use the actual Original Model output `out_df_ori` for the "Teacher" in difference vector calculation,
    # because `out_ori_random` is a specialized target for MD, but the "state" of the model is best captured by its reaction to the input.
    # However, MD is focusing on the "mismatched" task. 
    # If the user wants a1 to control MD loss, maybe the difference vectors should reflect the MD inputs?
    # MD Inputs: Student=out_df_un["fusion"], Teacher=func_max(out_ori_random)
    # Let's use these for consistency with the loss term.
    
    # MD Inputs for Difference Vector:
    # Student: out_df_un["fusion_logits"]
    # Teacher: func_max(out_ori_random, forget_labels) -> this is the target used in loss.
    # GT: gt_one_hot
    vec_md = get_diff_vectors(
        out_df_un["fusion_logits"], 
        func_max(out_ori_random, forget_labels), 
        gt_one_hot
    ) # [B, 60]
    
    # MKR Inputs for Difference Vector:
    # MKR uses Retain Set (batch_dr).
    # Student: out_dr_un["fusion"], out_dr_un["text"], out_dr_un["image"]
    # Teacher: func_max2(out_dr_ori) -> target used in loss.
    # GT: batch_dr["label"] -> need one-hot for retain.
    gt_one_hot_retain = F.one_hot(batch_dr["label"], num_classes=20).float()
    
    # Note: MKR is sum of 3 terms.
    # Target is same for all (func_max2(out_dr_ori)).
    # Student varies.
    vec_mkr_fusion = get_diff_vectors(out_dr_un["fusion_logits"], func_max2(out_dr_ori), gt_one_hot_retain)
    vec_mkr_text = get_diff_vectors(out_dr_un["text_logits"], func_max2(out_dr_ori), gt_one_hot_retain)
    vec_mkr_image = get_diff_vectors(out_dr_un["image_logits"], func_max2(out_dr_ori), gt_one_hot_retain)
    vec_mkr = torch.cat([vec_mkr_fusion, vec_mkr_text, vec_mkr_image], dim=1) # [B, 180]
    
    # UKR Inputs for Difference Vector:
    # UKR uses Forget Set (batch_df) but preserves Uni-modal performance.
    # Student: out_df_un["text"], out_df_un["image"]
    # Teacher: func_max(out_df_ori) -> target used in loss.
    # GT: gt_one_hot
    vec_ukr_text = get_diff_vectors(out_df_un["text_logits"], func_max(out_df_ori, forget_labels), gt_one_hot)
    vec_ukr_image = get_diff_vectors(out_df_un["image_logits"], func_max(out_df_ori, forget_labels), gt_one_hot)
    vec_ukr = torch.cat([vec_ukr_text, vec_ukr_image], dim=1) # [B, 120]

    # Predict weights
    # Note: MD/UKR are on Forget Set (batch_df), MKR on Retain Set (batch_dr).
    # They have same batch size. We can treat them as paired for the step or just take mean.
    # The coefficients need to be calculated per sample?
    # Wait, if we combine MD (on df) and MKR (on dr), how do we balance them per sample?
    # Usually `total_loss = a1*MD + a2*MKR + a3*UKR`.
    # If the batches are separate, we can't have a single "per sample" a1, a2, a3 set if the samples correspond to different things (forget vs retain).
    # But here we are summing losses.
    # Strategy: Compute `a1` for Forget Sample `i`, `average(a2)` for the batch?
    # Or should we assume the weights are global for the batch?
    # "The 3 a1,a2 and a3 values would be from 3 different neural networks."
    # "total_loss = a1.loss_md + a2.loss_multi + a3.loss_uni"
    # If a1 is [B_df, 1], MD is [B_df]. a1*MD is element-wise.
    # If a2 is [B_dr, 1], MKR is [B_dr]. a2*MKR is element-wise.
    # We can handle this by computing per-sample weighted losses then taking the mean.
    
    # HOWEVER: To normalize [a1, a2, a3] via softmax, they must be aligned.
    # We have `raw_a1` [B, 1] (from forget data), `raw_a2` [B, 1] (from retain data?), `raw_a3` [B, 1] (from forget data).
    # They are not aligned samples.
    # But we want to balance the TASKS.
    # The weights probably shouldn't be per-sample if the samples are disjoint sets (Modify vs Retain).
    # But the Prompt says: "Each of these would be a 1x10 vector... input to it's neural network to get a1". This implies per-sample input.
    # If I just Average the raw scores over the batch, then normalize?
    # Or: `a1_i` adjusts `Loss_MD_i`. `a2_j` adjusts `Loss_MKR_j`.
    # But then how to use Softmax([a1, a2, a3])? 
    # If they are different samples, element-wise comparison is meaningless.
    # COMPROMISE:
    # 1. Compute `raw_a1` for all samples in batch_df. Mean it -> `A1_score`.
    # 2. Compute `raw_a3` for all samples in batch_df. Mean it -> `A3_score`.
    # 3. Compute `raw_a2` for all samples in batch_dr. Mean it -> `A2_score`.
    # 4. Softmax([A1_score, A2_score, A3_score]) * 3 -> [Alpha1, Alpha2, Alpha3].
    # 5. Use these scalar weights for the batch mean losses.
    # This fulfills "datadriven loss function" and "networks determine weights" while solving the batch alignment issue.
    
    raw_a1 = net_a1(vec_md).mean()
    raw_a2 = net_a2(vec_mkr).mean()
    raw_a3 = net_a3(vec_ukr).mean()
    
    scores = torch.stack([raw_a1, raw_a2, raw_a3]) # [3]
    weights = F.softmax(scores, dim=0) * 3
    a1, a2, a3 = weights[0], weights[1], weights[2]

    # Calculate Losses (reduction='batchmean' effectively, but weighted)
    # Since we are using global scalar weights for the batch, we can use the default reduction='batchmean' functions 
    # (actually we updated them to support 'none', so let's use 'batchmean' to get scalars).
    # Wait, I updated them. 'batchmean' returns scalar.
    
    loss_md_val = MD_loss(func_max(out_ori_random,forget_labels), out_df_un["fusion_logits"], T, forget_labels, reduction="batchmean")
    
    loss_multi_val = (
        MKR_loss(func_max2(out_dr_ori), out_dr_un["fusion_logits"], T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max2(out_dr_ori), out_dr_un["image_logits"], T, forget_labels, reduction="batchmean") +
        MKR_loss(func_max2(out_dr_ori), out_dr_un["text_logits"], T, forget_labels, reduction="batchmean")
    ) / 3.0
    
    loss_uni_val = 0.5 * (
        UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["text_logits"], T, reduction="batchmean") +
        UKR_loss(func_max(out_df_ori,forget_labels), out_df_un["image_logits"], T, reduction="batchmean")
    )

    total_loss = (a1 * loss_md_val) + (a2 * loss_multi_val) + (a3 * loss_uni_val)

    return {
        "train_loss": total_loss,
        "loss_md": loss_md_val,
        "loss_multi": loss_multi_val,
        "loss_uni": loss_uni_val,
        "a1": a1.item(),
        "a2": a2.item(),
        "a3": a3.item()
    }



for epoch in range(EPOCHS):
    for batch_df, batch_dr in zip(forget_loader, retain_loader):

        optimizer.zero_grad()

        out = unlearning_train_step(
            model_ori,
            model_unlearn,
            net_a1, net_a2, net_a3,
            batch_df,
            batch_dr
        )

        out["train_loss"].backward()
        optimizer.step()

    print(
        f"Epoch {epoch} | "
        f"Total {out['train_loss'].item():.4f} | "
        f"MD {out['loss_md'].item():.4f} | "
        f"Multi {out['loss_multi'].item():.4f} | "
        f"Uni {out['loss_uni'].item():.4f} | "
        f"a1 {out['a1']:.2f} | a2 {out['a2']:.2f} | a3 {out['a3']:.2f}"
    )

    # print(
    #     f"Epoch {epoch} | "
    #     f"Total {out['train_loss'].item():.4f} | "
    #     f"MD {out['loss_md'].item():.4f} | "
    #     f"Multi {out['loss_multi'].item():.4f} | "
    #     f"Uni {out['loss_uni'].item():.4f}"
    # )

torch.save(model_unlearn.state_dict(), UNLEARNED_MODEL_PATH)
print("Unlearning complete. Model saved.")