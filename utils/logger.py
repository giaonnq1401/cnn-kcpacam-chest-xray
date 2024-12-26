# logger.py

import pandas as pd
from datetime import datetime

class ExperimentLogger:
   def __init__(self, excel_path):
       self.excel_path = excel_path
       self.results = []
       self.today = datetime.now().strftime('%Y-%m-%d')
       
       try:
           self.df = pd.read_excel(excel_path)
           if 'date' in self.df.columns and 'model' in self.df.columns:
               self.df = self.df[~(self.df['date'] == self.today)]
       except (FileNotFoundError, pd.errors.EmptyDataError):
           self.df = pd.DataFrame()

   def log_epoch(self, model_name, epoch, train_loss, val_loss, train_metrics, val_metrics, classes):
       row = {
           'date': self.today,
           'model': model_name,
           'epoch': epoch,
           'train_loss': train_loss,
           'val_loss': val_loss,
           'train_macro_recall': train_metrics['macro']['recall'],
           'train_macro_precision': train_metrics['macro']['precision'],
           'train_macro_f1': train_metrics['macro']['f1'],
           'train_macro_auc': train_metrics['macro']['auc'],
           'val_macro_recall': val_metrics['macro']['recall'],
           'val_macro_precision': val_metrics['macro']['precision'],
           'val_macro_f1': val_metrics['macro']['f1'],
           'val_macro_auc': val_metrics['macro']['auc']
       }
       
       for i, class_name in enumerate(classes):
           for metric in ['recall', 'precision', 'f1', 'auc']:
               row[f'{class_name}_{metric}'] = val_metrics['per_class'][metric][i]
       
       self.results.append(row)

   def save_results(self):
       new_df = pd.DataFrame(self.results)
       if len(self.df) > 0:
           self.df = pd.concat([self.df, new_df], ignore_index=True)
       else:
           self.df = new_df
       self.df.to_excel(self.excel_path, index=False)