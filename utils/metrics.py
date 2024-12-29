# metrics.py

import numpy as np
from sklearn.metrics import recall_score, roc_auc_score

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
        # Input validation
        if y_true.shape != y_pred_proba.shape:
            raise ValueError(f"Shape mismatch: y_true {y_true.shape} != y_pred_proba {y_pred_proba.shape}")

        # Convert probabilities to binary predictions using threshold
        y_pred = (y_pred_proba >= self.threshold).astype(int)
        
        # Initialize metrics dict
        metrics = {
            'overall': {},
            'per_class': {
                'recall': [],
                'auc': [],
                'n_positive': [],  # Number of positive samples per class
                'mean_prob': []    # Mean probability for positive samples
            }
        }
        
        # Calculate per-class metrics
        n_classes = y_true.shape[1]
        for i in range(n_classes):
            # Get current class data
            y_true_i = y_true[:, i]
            y_pred_i = y_pred[:, i]
            y_pred_proba_i = y_pred_proba[:, i]
            
            # Count positive samples
            n_positive = np.sum(y_true_i)
            metrics['per_class']['n_positive'].append(int(n_positive))
            
            # Calculate mean probability for positive samples
            pos_mask = y_true_i == 1
            mean_prob = np.mean(y_pred_proba_i[pos_mask]) if n_positive > 0 else 0
            metrics['per_class']['mean_prob'].append(float(mean_prob))

            # Skip metrics calculation if no positive samples
            if n_positive == 0:
                metrics['per_class']['recall'].append(0.0)
                metrics['per_class']['auc'].append(0.5)
                continue
            
            # Calculate metrics for this class
            try:
                recall = recall_score(y_true_i, y_pred_i, zero_division=0)
                auc = roc_auc_score(y_true_i, y_pred_proba_i)
                
                metrics['per_class']['recall'].append(float(recall))
                metrics['per_class']['auc'].append(float(auc))
                
            except Exception as e:
                print(f"Error calculating metrics for class {i}: {str(e)}")
                print(f"Number of positive samples: {n_positive}")
                print(f"Mean probability: {mean_prob:.4f}")
                metrics['per_class']['recall'].append(0.0)
                metrics['per_class']['auc'].append(0.5)
        
        # Calculate overall metrics (macro average)
        metrics['overall']['recall'] = float(np.mean(metrics['per_class']['recall']))
        metrics['overall']['auc'] = float(np.mean(metrics['per_class']['auc']))
        
        # Add overall statistics
        metrics['overall']['total_positive'] = int(np.sum(y_true))
        metrics['overall']['total_predictions'] = int(np.sum(y_pred))
        
        return metrics

    def _safe_divide(self, numerator, denominator):
        """Safe division handling zero denominator"""
        return np.divide(numerator, denominator, 
                        out=np.zeros_like(numerator, dtype=float), 
                        where=denominator!=0)