import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader,Subset
import random
from sklearn.metrics import (
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score
)

from src.dataset import Food101Dataset
from src.model_new_r import MultimodalFoodClassifier


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

BATCH_SIZE = 8
NUM_CLASSES = 101

TEST_CSV = "/home/team2/Unlearning/newDirauth2/data/food101/annotations/test_titles_cup_cakes.csv" # test
IMAGE_ROOT = "/home/team2/Unlearning/newDirauth2/data/food101/images/test" # test
#IMAGE_ROOT = "/home/team2/Unlearning/newDir/data/food101/images/test"

MODEL_PATH = "/home/team2/Unlearning/newDirauth2/models/oracle_model.pth"
#MODEL_PATH = "/home/team2/Unlearning/newDirauth2/full_dataset/models/best_multimodal_food101.pth"

OUTPUT_DIR = "/home/team2/Unlearning/newDirauth2/outputs/oracle/forget"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD DA
# --------------------------------------------------
print("Loading test dataset...")
dataset = Food101Dataset(TEST_CSV, IMAGE_ROOT)



# generator = torch.Generator().manual_seed(42)  # reproducible
# indices = torch.randperm(len(dataset), generator=generator)[:700]

# dataset = Subset(dataset, indices)



loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

class_names = list(range(NUM_CLASSES))



# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading model...")
model = MultimodalFoodClassifier(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

def per_class_accuracy(y_true, y_pred, num_classes):
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    class_accuracy = [
        (class_correct[i] / class_total[i] * 100) if class_total[i] > 0 else 0.0
        for i in range(num_classes)
    ]
    return class_accuracy
# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------
def evaluate(use_image=True, use_text=True, tag="multimodal"):
    y_true, y_pred = [], []

    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(
                image=image,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_image=use_image,
                use_text=use_text,
                return_intermediate=True
            )

            fusion_logits, text_logits, image_logits = outputs["fusion_logits"], outputs["text_logits"], outputs["image_logits"]
            if use_image and use_text:
                logits = fusion_logits
            elif use_image:
                logits = image_logits
            elif use_text:
                logits = text_logits
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # ---------------- Metrics ----------------
    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=range(NUM_CLASSES),
        zero_division=0
    )

    metrics_df = pd.DataFrame({
        "Class": class_names,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Support": support
    })

    # ---------------- Confusion Matrix ----------------
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm.astype(float) / row_sums * 100

    # ---------------- Per-Class Accuracy ----------------
    class_accuracy = per_class_accuracy(y_true, y_pred, NUM_CLASSES)
    metrics_df["Accuracy"] = class_accuracy

    # ---------------- Save Outputs ----------------
    metrics_df.to_csv(
        f"{OUTPUT_DIR}/{tag}_classwise_metrics.csv",
        index=False
    )

    np.savetxt(
        f"{OUTPUT_DIR}/{tag}_confusion_matrix_normalized.csv",
        cm_norm,
        delimiter=",",
        fmt="%.2f"
    )

    with open(f"{OUTPUT_DIR}/{tag}_summary.txt", "w") as f:
        f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

    print(f"[{tag}] Accuracy: {accuracy:.4f}")

    return cm_norm, metrics_df, accuracy


# --------------------------------------------------
# RUN MODALITY-WISE EVALUATION
# --------------------------------------------------
print("\nEvaluating IMAGE-ONLY model...")
cm_image, df_image, acc_image = evaluate(
    use_image=True,
    use_text=False,
    tag="image_only"
)

print("\nEvaluating TEXT-ONLY model...")
cm_text, df_text, acc_text = evaluate(
    use_image=False,
    use_text=True,
    tag="text_only"
)

print("\nEvaluating MULTIMODAL model...")
cm_multi, df_multi, acc_multi = evaluate(
    use_image=True,
    use_text=True,
    tag="multimodal"
)

comparison_df = pd.DataFrame({
    "Class": class_names,
    "Text-Only Accuracy": df_text["Accuracy"],
    "Image-Only Accuracy": df_image["Accuracy"],
    "Multimodal Accuracy": df_multi["Accuracy"]
})

comparison_df.to_csv(f"{OUTPUT_DIR}/per_class_accuracy_comparison.csv", index=False)
print("\nPer-class accuracy comparison saved.")

# --------------------------------------------------
# PLOT CONFUSION MATRICES
# --------------------------------------------------
def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(14, 12))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title(title)
    plt.colorbar(format="%.0f%%")

    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=90)
    plt.yticks(ticks, classes)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


print("\nSaving confusion matrix plots...")

plot_confusion_matrix(
    cm_image,
    class_names,
    "Image-Only Confusion Matrix (%)",
    f"{OUTPUT_DIR}/image_only_confusion_matrix.png"
)

plot_confusion_matrix(
    cm_text,
    class_names,
    "Text-Only Confusion Matrix (%)",
    f"{OUTPUT_DIR}/text_only_confusion_matrix.png"
)

plot_confusion_matrix(
    cm_multi,
    class_names,
    "Multimodal Confusion Matrix (%)",
    f"{OUTPUT_DIR}/multimodal_confusion_matrix.png"
)

print("\nEvaluation complete. All results saved in:", OUTPUT_DIR)





# import os
# import torch
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from torch.utils.data import DataLoader,Subset
# import random
# from sklearn.metrics import (
#     confusion_matrix,
#     precision_recall_fscore_support,
#     accuracy_score
# )

# from src.dataset import Food101Dataset
# from src.model import MultimodalFoodClassifier


# # --------------------------------------------------
# # CONFIG
# # --------------------------------------------------
# DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

# BATCH_SIZE = 8
# NUM_CLASSES = 20

# TEST_CSV = "/home/team2/Unlearning/newDirauth2/data/food101/annotations/test_titles_20classes.csv" # test
# IMAGE_ROOT = "/home/team2/Unlearning/newDirauth2/data/food101/images/test" # test
# # IMAGE_ROOT = "/home/team2/Unlearning/newDir/data/food101/images/test"

# #MODEL_PATH = "/home/team2/Unlearning/newDirauth2/newArch/models1/best_multimodal_food101_20cls.pth"
# MODEL_PATH = "/home/team2/Unlearning/newDirauth2/models5/best_multimodal_food101_101cls.pth"

# OUTPUT_DIR = "/home/team2/Unlearning/newDirauth2/output6/learn_logit_random"
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # --------------------------------------------------
# # LOAD DATA
# # --------------------------------------------------
# print("Loading test dataset...")
# dataset = Food101Dataset(TEST_CSV, IMAGE_ROOT)



# # generator = torch.Generator().manual_seed(42)  # reproducible
# # indices = torch.randperm(len(dataset), generator=generator)[:700]

# # dataset = Subset(dataset, indices)



# loader = DataLoader(
#     dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True,
# )

# class_names = list(range(NUM_CLASSES))



# # --------------------------------------------------
# # LOAD MODEL
# # --------------------------------------------------
# print("Loading model...")
# model = MultimodalFoodClassifier(num_classes=NUM_CLASSES)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)
# model.eval()


# # --------------------------------------------------
# # EVALUATION FUNCTION
# # --------------------------------------------------
# def evaluate(use_image=True, use_text=True, tag="multimodal"):
#     y_true, y_pred = [], []

#     with torch.no_grad():
#         for batch in loader:
#             image = batch["image"].to(DEVICE)
#             input_ids = batch["input_ids"].to(DEVICE)
#             attention_mask = batch["attention_mask"].to(DEVICE)
#             labels = batch["label"].to(DEVICE)

#             outputs = model(
#                 image=image,
#                 input_ids=input_ids,
#                 attention_mask=attention_mask,
#                 use_image=use_image,
#                 use_text=use_text,
#                 return_intermediate=True
#             )

#             # fusion_logits, text_logits, image_logits = outputs["fusion_logits"], outputs["text_logits"], outputs["image_logits"]
#             # if use_image and use_text:
#             #     logits = fusion_logits
#             # elif use_image:
#             #     logits = image_logits
#             # elif use_text:
#             #     logits = text_logits

#             logits=outputs["logits"]

#             preds = torch.argmax(logits, dim=1)

#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.cpu().numpy())

#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)

#     # ---------------- Metrics ----------------
#     accuracy = accuracy_score(y_true, y_pred)

#     precision, recall, f1, support = precision_recall_fscore_support(
#         y_true,
#         y_pred,
#         labels=range(NUM_CLASSES),
#         zero_division=0
#     )

#     metrics_df = pd.DataFrame({
#         "Class": class_names,
#         "Precision": precision,
#         "Recall": recall,
#         "F1-Score": f1,
#         "Support": support
#     })

#     # ---------------- Confusion Matrix ----------------
#     cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
#     row_sums = cm.sum(axis=1, keepdims=True)
#     row_sums[row_sums == 0] = 1
#     cm_norm = cm.astype(float) / row_sums * 100


#     # ---------------- Save Outputs ----------------
#     metrics_df.to_csv(
#         f"{OUTPUT_DIR}/{tag}_classwise_metrics.csv",
#         index=False
#     )

#     np.savetxt(
#         f"{OUTPUT_DIR}/{tag}_confusion_matrix_normalized.csv",
#         cm_norm,
#         delimiter=",",
#         fmt="%.2f"
#     )

#     with open(f"{OUTPUT_DIR}/{tag}_summary.txt", "w") as f:
#         f.write(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")

#     print(f"[{tag}] Accuracy: {accuracy:.4f}")

#     return cm_norm, metrics_df, accuracy


# # --------------------------------------------------
# # RUN MODALITY-WISE EVALUATION
# # --------------------------------------------------
# print("\nEvaluating IMAGE-ONLY model...")
# cm_image, df_image, acc_image = evaluate(
#     use_image=True,
#     use_text=False,
#     tag="image_only"
# )

# print("\nEvaluating TEXT-ONLY model...")
# cm_text, df_text, acc_text = evaluate(
#     use_image=False,
#     use_text=True,
#     tag="text_only"
# )

# print("\nEvaluating MULTIMODAL model...")
# cm_multi, df_multi, acc_multi = evaluate(
#     use_image=True,
#     use_text=True,
#     tag="multimodal"
# )


# # --------------------------------------------------
# # PLOT CONFUSION MATRICES
# # --------------------------------------------------
# def plot_confusion_matrix(cm, classes, title, save_path):
#     plt.figure(figsize=(14, 12))
#     plt.imshow(cm, interpolation="nearest", cmap="viridis")
#     plt.title(title)
#     plt.colorbar(format="%.0f%%")

#     ticks = np.arange(len(classes))
#     plt.xticks(ticks, classes, rotation=90)
#     plt.yticks(ticks, classes)

#     plt.xlabel("Predicted Label")
#     plt.ylabel("True Label")

#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches="tight")
#     plt.close()


# print("\nSaving confusion matrix plots...")

# plot_confusion_matrix(
#     cm_image,
#     class_names,
#     "Image-Only Confusion Matrix (%)",
#     f"{OUTPUT_DIR}/image_only_confusion_matrix.png"
# )

# plot_confusion_matrix(
#     cm_text,
#     class_names,
#     "Text-Only Confusion Matrix (%)",
#     f"{OUTPUT_DIR}/text_only_confusion_matrix.png"
# )

# plot_confusion_matrix(
#     cm_multi,
#     class_names,
#     "Multimodal Confusion Matrix (%)",
#     f"{OUTPUT_DIR}/multimodal_confusion_matrix.png"
# )

# print("\nEvaluation complete. All results saved in:", OUTPUT_DIR)
