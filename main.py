import os
import torch
from torchvision.models import resnet50, ResNet50_Weights
from utils.preprocess import load_chest_xray_dataset
from utils.visualize import visualize_cam
from pytorch_grad_cam import KPCA_CAM
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

def initialize_model(device):
    """
    Khởi tạo mô hình ResNet50 và đưa về thiết bị CPU/GPU.
    Args:
        device (str): Thiết bị (CPU hoặc GPU).
    Returns:
        model (torch.nn.Module): Mô hình ResNet50 đã được khởi tạo.
    """
    # model = resnet50(pretrained=True)
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 lớp: Normal, Pneumonia
    model.to(device)
    return model

def fine_tune_model(model, train_dataset, val_dataset, device, epochs=5, batch_size=32):
    """
    Fine-tune mô hình ResNet50 trên tập X-ray phổi.
    Args:
        model (torch.nn.Module): Mô hình đã load.
        train_dataset (list): Tập train (ảnh và label).
        val_dataset (list): Tập validation (ảnh và label).
        device (str): Thiết bị (CPU hoặc GPU).
        epochs (int): Số epoch huấn luyện.
        batch_size (int): Kích thước batch.
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        print(f"Validation Accuracy: {correct / total * 100:.2f}%")

def perform_kpca_cam(model, image_tensor, device, output_path):
    """
    Tính toán KPCA-CAM trên một ảnh X-ray.
    Args:
        model (torch.nn.Module): Mô hình ResNet50 đã fine-tune.
        image_tensor (torch.Tensor): Tensor của ảnh đầu vào.
        device (str): Thiết bị (CPU hoặc GPU).
        output_path (str): Đường dẫn lưu kết quả KPCA-CAM.
    """
    target_layer = model.layer4[-1]
    cam = KPCA_CAM(model=model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=image_tensor.unsqueeze(0).to(device), targets=None)

    visualize_cam(image_tensor.cpu(), grayscale_cam[0], output_path)

def main():
    """
    Chương trình chính để fine-tune mô hình và thực hiện KPCA-CAM.
    """
    # Cấu hình thiết bị
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tải dataset
    print("Loading dataset...")
    train_dataset = load_chest_xray_dataset(split="train")
    val_dataset = load_chest_xray_dataset(split="validation")
    test_dataset = load_chest_xray_dataset(split="test")

    # Khởi tạo và fine-tune mô hình
    print("Initializing and fine-tuning model...")
    model = initialize_model(device)
    fine_tune_model(model, train_dataset, val_dataset, device, epochs=2, batch_size=16)

    # Lấy một ảnh từ tập test và thực hiện KPCA-CAM
    print("Performing KPCA-CAM...")
    test_image, _ = test_dataset[0]  # Lấy ảnh đầu tiên từ tập test
    output_path = os.path.join("outputs", "kpca_cam_test_result.jpg")
    perform_kpca_cam(model, test_image, device, output_path)

    print(f"KPCA-CAM result saved to {output_path}")

if __name__ == "__main__":
    main()
