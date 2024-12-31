import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, df, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.df = df
        
        # Validate image existence
        valid_imgs = []
        for idx in range(len(df)):
            img_path = self.data_dir / 'images' / 'images_002' / 'images' / df.iloc[idx]['Image Index']
            if img_path.exists():
                valid_imgs.append(idx)
            
        self.df = df.iloc[valid_imgs].reset_index(drop=True)
        logging.info(f"Found {len(self.df)} valid images out of {len(df)} total images")
        
        if len(self.df) == 0:
            raise ValueError("No valid images found in the dataset directory")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.data_dir / 'images' / 'images_002' / 'images' / self.df.iloc[idx]['Image Index']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        label = 0 if self.df.iloc[idx]['No Finding'] == 1 else 1
        return image, torch.tensor(label, dtype=torch.float32)

def prepare_data(data_dir, batch_size=32, num_train=None, num_test=None):
    try:
        data_dir = Path(data_dir)
        csv_path = data_dir / 'data_Data_Entry_2017_v2020.csv'
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found at {csv_path}")
            
        # Read main CSV
        df = pd.read_csv(csv_path)
        
        # Create binary labels
        df['No Finding'] = df['Finding Labels'].apply(lambda x: 1 if 'No Finding' in x else 0)
        
        # Split into train/val/test
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
        
        # Limit samples if specified
        if num_train:
            train_df = train_df[:num_train]
        if num_test:
            test_df = test_df[:num_test]
            
        # Reset indices
        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        
        logging.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}, Test samples: {len(test_df)}")

        # Define transforms
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create datasets
        train_dataset = ChestXrayDataset(data_dir, train_df, train_transform)
        val_dataset = ChestXrayDataset(data_dir, val_df, test_transform)
        test_dataset = ChestXrayDataset(data_dir, test_df, test_transform)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        return train_loader, val_loader, test_loader

    except Exception as e:
        logging.error(f"Error in prepare_data: {str(e)}")
        raise

def get_resnet50():
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model

def get_vgg16():
    model = models.vgg16(weights='IMAGENET1K_V1')
    model.classifier[6] = nn.Sequential(
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model

def get_vit():
    # model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    # model.classifier = nn.Sequential(
    #     nn.Linear(768, 1),
    #     nn.Sigmoid()
    # )
    model = models.vit_b_16(weights='IMAGENET1K_V1')
    model.heads = nn.Sequential(
        nn.Linear(768, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return model

def train_model(model, train_loader, val_loader, epochs=5, device='cuda', learning_rate=1e-4):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 5
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_preds = []
        train_targets = []
        
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle different model outputs
            if isinstance(model, models.vision_transformer.VisionTransformer):
            # if isinstance(model, models.vit.VisionTransformer):
                outputs = outputs.squeeze()
            elif hasattr(outputs, 'logits'):
                outputs = outputs.logits.squeeze()
            else:
                outputs = outputs.squeeze()
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.cpu().detach().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                
                # Handle different model outputs
                # if isinstance(model, models.vit.VisionTransformer):
                if isinstance(model, models.vision_transformer.VisionTransformer):
                    outputs = outputs.squeeze()
                elif hasattr(outputs, 'logits'):
                    outputs = outputs.logits.squeeze()
                else:
                    outputs = outputs.squeeze()
                    
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_preds = np.array(train_preds) >= 0.5
        val_preds = np.array(val_preds) >= 0.5
        
        train_recall = recall_score(train_targets, train_preds)
        val_recall = recall_score(val_targets, val_preds)
        
        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train Recall: {train_recall:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Recall: {val_recall:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f'best_model_{model.__class__.__name__}.pth')
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {epoch + 1} epochs')
            break
    
    return model
# Usage
if __name__ == "__main__":
    data_dir = './dataset'  # Update this path to your dataset location
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        train_loader, val_loader, test_loader = prepare_data(
            data_dir=data_dir,
            batch_size=32,
            num_train=100,
            num_test=20
        )
        
        # Train ResNet50
        # print("Training ResNet50...")
        # resnet = get_resnet50()
        # train_model(resnet, train_loader, val_loader, device=device)
        
        # Train VGG16
        # print("Training VGG16...")
        # vgg = get_vgg16()
        # train_model(vgg, train_loader, val_loader, device=device)
        
        # Train ViT
        print("Training ViT...")
        vit = get_vit()
        train_model(vit, train_loader, val_loader, device=device)
        
    except Exception as e:
        logging.error(f"Training error: {str(e)}")