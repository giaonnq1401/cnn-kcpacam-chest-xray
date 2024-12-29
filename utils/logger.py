# logger.py

import pandas as pd
import os
from datetime import datetime
import numpy as np

class ExperimentLogger:
    def __init__(self, log_path):
        """
        Initialize the experiment logger
        Args:
            log_path (str): Path to save the Excel log file
        """
        self.log_path = log_path
        self.experiment_data = []
        self.detailed_metrics = {}
        
    def log_epoch(self, model_name, epoch, train_loss, val_loss, train_metrics, val_metrics, classes):
        """
        Log metrics for one epoch
        Args:
            model_name (str): Name of the model
            epoch (int): Current epoch number
            train_loss (float): Training loss
            val_loss (float): Validation loss
            train_metrics (dict): Training metrics dictionary
            val_metrics (dict): Validation metrics dictionary
            classes (list): List of class names
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log overall metrics
        overall_data = {
            'timestamp': timestamp,
            'model': model_name,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_recall': train_metrics['overall']['recall'],
            'train_f1': train_metrics['overall']['f1'],
            'train_auc': train_metrics['overall']['auc'],
            'val_recall': val_metrics['overall']['recall'],
            'val_f1': val_metrics['overall']['f1'],
            'val_auc': val_metrics['overall']['auc']
        }
        self.experiment_data.append(overall_data)
        
        # Log per-class metrics
        model_key = f"{model_name}_epoch{epoch}"
        self.detailed_metrics[model_key] = {
            'timestamp': timestamp,
            'classes': classes,
            'train_metrics': train_metrics['per_class'],
            'val_metrics': val_metrics['per_class']
        }

    def save_results(self):
        """
        Save all logged data to Excel file with multiple sheets
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        with pd.ExcelWriter(self.log_path, engine='openpyxl') as writer:
            # Save overall metrics
            overall_df = pd.DataFrame(self.experiment_data)
            overall_df.to_excel(writer, sheet_name='Overall_Metrics', index=False)
            
            # Save per-class metrics for each model and epoch
            for model_key, data in self.detailed_metrics.items():
                classes = data['classes']
                
                # Create DataFrame for training metrics
                train_metrics_data = {
                    'Class': classes,
                    'Recall': data['train_metrics']['recall'],
                    'F1': data['train_metrics']['f1'],
                    'AUC': data['train_metrics']['auc']
                }
                train_df = pd.DataFrame(train_metrics_data)
                
                # Create DataFrame for validation metrics
                val_metrics_data = {
                    'Class': classes,
                    'Recall': data['val_metrics']['recall'],
                    'F1': data['val_metrics']['f1'],
                    'AUC': data['val_metrics']['auc']
                }
                val_df = pd.DataFrame(val_metrics_data)
                
                # Save to Excel
                sheet_name = f'{model_key}'
                train_df.to_excel(writer, sheet_name=sheet_name, 
                                startrow=1, startcol=0, index=False)
                val_df.to_excel(writer, sheet_name=sheet_name,
                              startrow=len(classes)+4, startcol=0, index=False)
                
                # Write headers
                worksheet = writer.sheets[sheet_name]
                worksheet.cell(row=1, column=1, value="Training Metrics")
                worksheet.cell(row=len(classes)+4, column=1, 
                             value="Validation Metrics")

    def get_best_model_metrics(self):
        """
        Get metrics for the best performing model based on validation F1 score
        Returns:
            dict: Best model metrics
        """
        if not self.experiment_data:
            return None
            
        df = pd.DataFrame(self.experiment_data)
        best_idx = df['val_f1'].idxmax()
        return df.iloc[best_idx].to_dict()

    def plot_training_curves(self, save_path=None):
        """
        Plot training and validation metrics over epochs
        Args:
            save_path (str, optional): Path to save the plot
        """
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame(self.experiment_data)
            plt.figure(figsize=(15, 10))
            
            # Plot loss
            plt.subplot(2, 2, 1)
            plt.plot(df['epoch'], df['train_loss'], label='Train Loss')
            plt.plot(df['epoch'], df['val_loss'], label='Val Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot F1 Score
            plt.subplot(2, 2, 2)
            plt.plot(df['epoch'], df['train_f1'], label='Train F1')
            plt.plot(df['epoch'], df['val_f1'], label='Val F1')
            plt.title('F1 Score Curves')
            plt.xlabel('Epoch')
            plt.ylabel('F1 Score')
            plt.legend()
            
            # Plot Recall
            plt.subplot(2, 2, 4)
            plt.plot(df['epoch'], df['train_recall'], label='Train Recall')
            plt.plot(df['epoch'], df['val_recall'], label='Val Recall')
            plt.title('Recall Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Recall')
            plt.legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
            plt.show()
            
        except ImportError:
            print("matplotlib is required for plotting training curves")