# trainer.py

import torch
import numpy as np
from utils.metrics import MetricsCalculator

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
