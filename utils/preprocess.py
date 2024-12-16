import os
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

def preprocess_image(image_path):
    """
    Preprocess an image: resize, convert to tensor, and normalize.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to 224x224
        transforms.ToTensor(),         # Convert image to tensor
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Normalization values for ImageNet
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB
    return transform(image)

def encode_label(label):
    """
    Encode labels into numerical format.
    'Normal' -> 0, 'Pneumonia' -> 1
    """
    return 0 if label == "Normal" else 1

class ChestXrayDataset(Dataset):
    """
    Custom dataset for Chest X-ray classification.
    """
    def __init__(self, dataset_split):
        self.dataset = dataset_split

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        image = preprocess_image(data["image_file_path"])  # Process image
        label = encode_label(data["labels"])          # Encode label
        return image, torch.tensor(label, dtype=torch.long)

def load_chest_xray_dataset(split: str = "train"):
    from datasets import load_dataset
    import torch
    from torchvision import transforms

    # Tải dataset từ Hugging Face
    dataset = load_dataset("keremberke/chest-xray-classification", name="full")
    subset = dataset[split]  # Lấy phần train, test, hoặc validation

    # Hàm tiền xử lý ảnh
    def preprocess_image(example):
        image = example["image"]
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(image)

    # Mã hóa nhãn
    # Nếu labels là số nguyên, ánh xạ không cần thiết
    label_mapping = {0: 0, 1: 1}  # Chỉ giữ nguyên nếu nhãn đã là số
    processed_data = []
    for example in subset:
        image_tensor = preprocess_image(example)
        label = example["labels"]  # Nếu labels là số, không cần ánh xạ
        processed_data.append((image_tensor, label))

    return processed_data
