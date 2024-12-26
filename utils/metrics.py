from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

class MetricsCalculator:
   @staticmethod
   def calculate_metrics(y_true, y_pred, threshold=0.5):
       y_pred_binary = (y_pred > threshold).astype(int)
       
       recalls = recall_score(y_true, y_pred_binary, average=None, zero_division=1)
       precisions = precision_score(y_true, y_pred_binary, average=None, zero_division=1) 
       f1_scores = f1_score(y_true, y_pred_binary, average=None, zero_division=1)
       
       # Handle the case where there are no positive predictions for some classes
       try:
           auc_scores = roc_auc_score(y_true, y_pred, average=None)
       except:
           auc_scores = [0] * y_true.shape[1]
           
       macro_recall = recall_score(y_true, y_pred_binary, average='macro', zero_division=1)
       macro_precision = precision_score(y_true, y_pred_binary, average='macro', zero_division=1)
       macro_f1 = f1_score(y_true, y_pred_binary, average='macro', zero_division=1)
       
       try:
           macro_auc = roc_auc_score(y_true, y_pred, average='macro')
       except:
           macro_auc = 0
           
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