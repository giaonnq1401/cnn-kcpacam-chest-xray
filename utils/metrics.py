from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score
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
