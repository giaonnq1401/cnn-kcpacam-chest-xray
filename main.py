# main.py

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from models.model_factory import ModelFactory
from utils.preprocess import ChestXrayDataset, get_transforms, prepare_data
from utils.logger import ExperimentLogger
from utils.trainer import ModelTrainer

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--csv_path', type=str, required=True)
   parser.add_argument('--img_dir', type=str, required=True)
   parser.add_argument('--train_list', type=str, required=True)
   parser.add_argument('--test_list', type=str, required=True)
   parser.add_argument('--model_name', type=str, default='resnet50', 
                     choices=['resnet50', 'vgg16', 'vit'])
   parser.add_argument('--batch_size', type=int, default=32)
   parser.add_argument('--num_epochs', type=int, default=10)
   parser.add_argument('--lr', type=float, default=0.001)
   parser.add_argument('--train_size', type=int, default=None, 
                   help='Number of training images to use. If None, use all available')
   parser.add_argument('--test_size', type=int, default=None,
                   help='Number of test images to use. If None, use all available')
   parser.add_argument('--output_dir', type=str, default='./checkpoints')
   args = parser.parse_args()

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
   train_df, test_df = prepare_data(args.csv_path, args.img_dir, 
                               args.train_list, args.test_list,
                               args.train_size, args.test_size)
   train_transforms, val_transforms = get_transforms()
   
   train_dataset = ChestXrayDataset(train_df, args.img_dir, train_transforms)
   val_dataset = ChestXrayDataset(test_df, args.img_dir, val_transforms)
   
   train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                           shuffle=True, num_workers=4)
   val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                         shuffle=False, num_workers=4)
   
   model = ModelFactory.get_model(args.model_name, len(train_dataset.classes))
   model = model.to(device)
   
   criterion = nn.BCELoss()
   optimizer = optim.Adam(model.parameters(), lr=args.lr)
   trainer = ModelTrainer(model, train_loader, val_loader, criterion, 
                        optimizer, device, train_dataset.classes)
   
   model_output_dir = os.path.join(args.output_dir, args.model_name)
   os.makedirs(model_output_dir, exist_ok=True)
   
   logger = ExperimentLogger(os.path.join(args.output_dir, 'model_comparison.xlsx'))
   
   for epoch in range(args.num_epochs):
       print(f'\nEpoch {epoch+1}/{args.num_epochs}')
       print('-' * 50)
       
       train_loss, train_metrics = trainer.train_epoch()
       val_loss, val_metrics = trainer.validate()
       
       logger.log_epoch(
           model_name=args.model_name,
           epoch=epoch+1,
           train_loss=train_loss,
           val_loss=val_loss,
           train_metrics=train_metrics,
           val_metrics=val_metrics,
           classes=trainer.classes
       )
       
       print('\nPer-class Metrics:')
       for i, class_name in enumerate(trainer.classes):
           print(f"\n{class_name}:")
           print(f"Recall: {val_metrics['per_class']['recall'][i]:.4f}")
           print(f"F1: {val_metrics['per_class']['f1'][i]:.4f}")
           print(f"AUC: {val_metrics['per_class']['auc'][i]:.4f}")
       
       print(f"Recall: {val_metrics['overall']['recall']}")
       print(f"F1: {val_metrics['overall']['f1']}")
       print(f"AUC: {val_metrics['overall']['auc']}")

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
   
   logger.save_results()

if __name__ == '__main__':
   main()