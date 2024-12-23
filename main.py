import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import (
    resnet50, ResNet50_Weights,
    vgg16, VGG16_Weights,
    vit_b_16, ViT_B_16_Weights,
)
import argparse
import os
from utils.preprocess import ChestXrayDataset, get_transforms, prepare_data
from utils.logger import ExperimentLogger
from utils.trainer import ModelTrainer

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