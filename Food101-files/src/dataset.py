import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from transformers import BertTokenizer
from src.label import label_to_idx, idx_to_label, LABEL_MAP
#from label import label_to_idx, idx_to_label, LABEL_MAP
#from label import label_to_idx, idx_to_label, LABEL_MAP


class Food101Dataset(Dataset):
    """
    Multimodal Food101 Dataset (Image + Text).
    Supports any number of classes (20 / 101).
    """

    def __init__(self, csv_path, image_root, max_length=64):
        self.data = pd.read_csv(csv_path, header=None)
        self.data.columns = ["image", "title", "label"]
        self.data["label"] = self.data["label"].astype(str)
        # Label mapping (automatic)
        # self.labels = sorted(self.data["label"].unique())
        # self.label2idx = {label: idx for idx, label in enumerate(self.labels)}
        self.label2idx = LABEL_MAP

        self.image_root = image_root
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Image preprocessing (ResNet-compatible)
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # -------- Image --------
        image_path = os.path.join(
            self.image_root,
            row["label"],
            row["image"]
        )

        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        # -------- Text --------
        text = str(row["title"])
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # -------- Label --------
        label = self.label2idx[row["label"]]

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }




# def create_train_val_split(csv_path, image_root, val_ratio=0.2, seed=42):
#     """
#     Creates deterministic train/val split from training CSV.
#     """
#     full_dataset = Food101Dataset(csv_path, image_root)

#     val_size = int(len(full_dataset) * val_ratio)
#     train_size = len(full_dataset) - val_size

#     generator = torch.Generator().manual_seed(seed)

#     train_dataset, val_dataset = random_split(
#         full_dataset,
#         [train_size, val_size],
#         generator=generator
#     )

#     return train_dataset, val_dataset, full_dataset.labels


def create_train_val_split(csv_path, image_root, val_ratio=0.2, seed=42):
    full_dataset = Food101Dataset(csv_path, image_root)

    val_size = int(len(full_dataset) * val_ratio)
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=generator
    )

    labels = list(full_dataset.label2idx.keys())
    return train_dataset, val_dataset, labels


def ForgetDataset():
    csv_path="/home/team2/Unlearning/Food101-files/data/annotations/test_titles_cup_cakes.csv"
    image_root="/home/team2/Unlearning/Food101-files/data/images/images/train"
    max_length=64
    return Food101Dataset(csv_path, image_root, max_length=max_length)

def RetainDataset():
    csv_path="/home/team2/Unlearning/Food101-files/data/annotations/test_titles_others_101.csv"
    image_root="/home/team2/Unlearning/Food101-files/data/images/images/train"
    max_length=64
    return Food101Dataset(csv_path, image_root, max_length=max_length)








# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# import pandas as pd
# import os
# from transformers import BertTokenizer
# from torchvision import transforms

# class Food101Dataset(Dataset):
#     def __init__(self, csv_path, image_root):
#         self.data = pd.read_csv(csv_path, header=None)
#         self.data.columns = ["image", "title", "label"]

#         # Label → index mapping
#         self.labels = sorted(self.data["label"].unique())
#         self.label2idx = {l: i for i, l in enumerate(self.labels)}

#         # Tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#         # Image transforms
#         self.image_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(
#                 mean=[0.485, 0.456, 0.406],
#                 std=[0.229, 0.224, 0.225]
#             )
#         ])

#         self.image_root = image_root
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self,idx):
#         row=self.data.iloc[idx]
#         image_path = os.path.join(
#             self.image_root,
#             row["label"],      # class folder
#             row["image"]       # image file
#         )

#         image = Image.open(image_path).convert("RGB")
#         image = self.image_transform(image)

#         text = str(row["title"])
#         encoding = self.tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=32,
#             return_tensors="pt"
#         )
#         label = self.label2idx[row["label"]]

#         return {
#             "image": image,
#             "input_ids": encoding["input_ids"].squeeze(0),
#             "attention_mask": encoding["attention_mask"].squeeze(0),
#             "label": torch.tensor(label)
#         } 