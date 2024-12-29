# metrics.py

import numpy as np
from sklearn.metrics import  recall_score, f1_score, roc_auc_score

class MetricsCalculator:
    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def calculate_metrics(self, y_true, y_pred_proba):
        """
        Calculate metrics for multi-label classification
        Args:
            y_true: Ground truth labels (binary)
            y_pred_proba: Predicted probabilities
        Returns:
            dict: Dictionary containing metrics
        """
        # Convert probabilities to binary predictions using threshold
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Initialize metrics dict
        metrics = {
            'overall': {},
            'per_class': {
                'recall': [],
                'f1': [],
                'auc': []
            }
        }
        
        # Calculate per-class metrics
        n_classes = y_true.shape[1]
        for i in range(n_classes):
            # Skip if no positive samples in ground truth
            if np.sum(y_true[:, i]) == 0:
                metrics['per_class']['recall'].append(0.0)
                metrics['per_class']['f1'].append(0.0)
                metrics['per_class']['auc'].append(0.5)  # Default AUC for no prediction
                continue
                
            # Calculate metrics for this class
            try:
                metrics['per_class']['recall'].append(
                    recall_score(y_true[:, i], y_pred[:, i], zero_division=0))
                metrics['per_class']['f1'].append(
                    f1_score(y_true[:, i], y_pred[:, i], zero_division=0))
                # AUC is calculated on probabilities, not binary predictions
                metrics['per_class']['auc'].append(
                    roc_auc_score(y_true[:, i], y_pred_proba[:, i]))
            except Exception as e:
                print(f"Error calculating metrics for class {i}: {str(e)}")
                metrics['per_class']['recall'].append(0.0)
                metrics['per_class']['f1'].append(0.0)
                metrics['per_class']['auc'].append(0.5)
        
        # Calculate overall metrics (macro average)
        metrics['overall']['recall'] = np.mean(metrics['per_class']['recall'])
        metrics['overall']['f1'] = np.mean(metrics['per_class']['f1'])
        metrics['overall']['auc'] = np.mean(metrics['per_class']['auc'])
        
        return metrics

    def _safe_divide(self, numerator, denominator):
        """Safe division handling zero denominator"""
        return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator!=0)