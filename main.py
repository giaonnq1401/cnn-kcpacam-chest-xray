import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    vit_b_16, ViT_B_16_Weights,
)
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
import numpy as np
import argparse
import os
from utils.preprocess import ChestXrayDataset, get_transforms, prepare_data, get_class_weights
from utils.logger import ExperimentLogger

class ModelFactory:
    """Factory class để tạo các model khác nhau"""
    @staticmethod
    def get_model(model_name, num_classes):
        model_name = model_name.lower()
        if model_name == 'resnet50':
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        elif model_name == 'vgg16':
            model = vgg16(weights=VGG16_Weights.DEFAULT)
            num_ftrs = model.classifier[-1].in_features
            model.classifier[-1] = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        elif model_name == 'vit':
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
            num_ftrs = model.heads.head.in_features
            model.heads.head = nn.Sequential(
                nn.Linear(num_ftrs, num_classes),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        return model

class MetricsCalculator:
    """Class để tính toán các metrics"""
    @staticmethod
    def calculate_metrics(y_true, y_pred, threshold=0.5):
        # Convert predictions to binary using threshold
        y_pred_binary = (y_pred > threshold).astype(int)
        
        # Calculate metrics for each class
        recalls = recall_score(y_true, y_pred_binary, average=None)
        precisions = precision_score(y_true, y_pred_binary, average=None)
        f1_scores = f1_score(y_true, y_pred_binary, average=None)
        auc_scores = roc_auc_score(y_true, y_pred, average=None)
        
        # Calculate macro averages
        macro_recall = recall_score(y_true, y_pred_binary, average='macro')
        macro_precision = precision_score(y_true, y_pred_binary, average='macro')
        macro_f1 = f1_score(y_true, y_pred_binary, average='macro')
        macro_auc = roc_auc_score(y_true, y_pred, average='macro')
        
        return {
            'per_class': {
                'recall': recalls,
                'precision': precisions,
                'f1': f1_scores,
                'auc': auc_scores
            },
            'macro': {
                'recall': macro_recall,
                'precision': macro_precision,
                'f1': macro_f1,
                'auc': macro_auc
            }
        }

class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device, classes):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.classes = classes
        self.best_val_loss = float('inf')
        self.metrics_calculator = MetricsCalculator()

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

        # Convert lists to numpy arrays
        predictions = np.array(predictions)
        labels_list = np.array(labels_list)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(labels_list, predictions)
        epoch_loss = running_loss / len(self.train_loader)
        
        return epoch_loss, metrics

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

        # Convert lists to numpy arrays
        predictions = np.array(predictions)
        labels_list = np.array(labels_list)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_metrics(labels_list, predictions)
        val_loss = val_loss / len(self.val_loader)
        
        return val_loss, metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='resnet50', 
                      choices=['resnet50', 'vgg16', 'vit'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--output_dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare data
    train_df, val_df = prepare_data(args.csv_path, args.img_dir)
    train_transforms, val_transforms = get_transforms()
    
    # Create datasets
    train_dataset = ChestXrayDataset(train_df, args.img_dir, train_transforms)
    val_dataset = ChestXrayDataset(val_df, args.img_dir, val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = ModelFactory.get_model(args.model_name, len(train_dataset.classes))
    model = model.to(device)
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = ModelTrainer(model, train_loader, val_loader, criterion, optimizer, device, train_dataset.classes)
    
    # Create output directory with model name
    model_output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Initialize experiment logger
    logger = ExperimentLogger(os.path.join(args.output_dir, 'model_comparison.xlsx'))
    
    # Training loop
    for epoch in range(args.num_epochs):
        print(f'\nEpoch {epoch+1}/{args.num_epochs}')
        print('-' * 50)
        
        # Training phase
        train_loss, train_metrics = trainer.train_epoch()
        
        # Validation phase
        val_loss, val_metrics = trainer.validate()
        
        # Log results
        logger.log_epoch(
            model_name=args.model_name,
            epoch=epoch+1,
            train_loss=train_loss,
            val_loss=val_loss,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            classes=trainer.classes
        )
        
        # Print per-class metrics
        print('\nPer-class Metrics:')
        for i, class_name in enumerate(trainer.classes):
            print(f"\n{class_name}:")
            print(f"Recall: {val_metrics['per_class']['recall'][i]:.4f}")
            print(f"Precision: {val_metrics['per_class']['precision'][i]:.4f}")
            print(f"F1: {val_metrics['per_class']['f1'][i]:.4f}")
            print(f"AUC: {val_metrics['per_class']['auc'][i]:.4f}")
        
        # Save best model
        if val_loss < trainer.best_val_loss:
            trainer.best_val_loss = val_loss
            model_save_path = os.path.join(model_output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'model_name': args.model_name
            }, model_save_path)
            print(f"\nSaved best model to {model_save_path}")
    
    # Save final results
    logger.save_results()

if __name__ == '__main__':
    main()