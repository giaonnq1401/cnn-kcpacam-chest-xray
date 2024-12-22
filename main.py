import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import roc_auc_score
import numpy as np
import argparse
import os
from utils.preprocess import ChestXrayDataset, get_transforms, prepare_data, get_class_weights

def get_model(num_classes):
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, num_classes),
        nn.Sigmoid()
    )
    return model

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.best_val_loss = float('inf')

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        predictions = []
        labels_list = []

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels.float())
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            predictions.extend(outputs.cpu().detach().numpy())
            labels_list.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(self.train_loader)
        epoch_auc = roc_auc_score(np.array(labels_list), np.array(predictions), average='macro')
        
        return epoch_loss, epoch_auc

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        predictions = []
        labels_list = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels.float())
                
                val_loss += loss.item()
                predictions.extend(outputs.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())

        val_loss = val_loss / len(self.val_loader)
        val_auc = roc_auc_score(np.array(labels_list), np.array(predictions), average='macro')
        
        return val_loss, val_auc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    train_df, val_df = prepare_data(args.csv_path, args.img_dir)
    
    # Print information about the number of train/val samples
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    train_transforms, val_transforms = get_transforms()
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_df, args.img_dir, train_transforms)
    val_dataset = ChestXrayDataset(val_df, args.img_dir, val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = get_model(len(train_dataset.classes))
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = ModelTrainer(model, train_loader, val_loader, criterion, optimizer, device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.num_epochs):
        train_loss, train_auc = trainer.train_epoch()
        val_loss, val_auc = trainer.validate()
        
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print(f'Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}')
        
        # Save best model
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_auc': val_auc
            }, os.path.join(args.output_dir, 'best_model.pth'))

if __name__ == '__main__':
    main()