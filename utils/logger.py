import pandas as pd
from datetime import datetime

class ExperimentLogger:
    def __init__(self, output_file='model_comparison_results.xlsx'):
        self.output_file = output_file
        self.results = {
            'timestamp': [],
            'model_name': [],
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_macro_recall': [],
            'train_macro_precision': [],
            'train_macro_f1': [],
            'train_macro_auc': [],
            'val_macro_recall': [],
            'val_macro_precision': [],
            'val_macro_f1': [],
            'val_macro_auc': [],
        }
        # Dictionary để lưu per-class metrics
        self.per_class_results = {}

    def log_epoch(self, model_name, epoch, train_loss, val_loss, train_metrics, val_metrics, classes):
        # Log basic metrics
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.results['timestamp'].append(timestamp)
        self.results['model_name'].append(model_name)
        self.results['epoch'].append(epoch)
        self.results['train_loss'].append(train_loss)
        self.results['val_loss'].append(val_loss)
        
        # Log macro metrics
        self.results['train_macro_recall'].append(train_metrics['macro']['recall'])
        self.results['train_macro_precision'].append(train_metrics['macro']['precision'])
        self.results['train_macro_f1'].append(train_metrics['macro']['f1'])
        self.results['train_macro_auc'].append(train_metrics['macro']['auc'])
        
        self.results['val_macro_recall'].append(val_metrics['macro']['recall'])
        self.results['val_macro_precision'].append(val_metrics['macro']['precision'])
        self.results['val_macro_f1'].append(val_metrics['macro']['f1'])
        self.results['val_macro_auc'].append(val_metrics['macro']['auc'])
        
        # Log per-class metrics
        for i, class_name in enumerate(classes):
            for metric in ['recall', 'precision', 'f1', 'auc']:
                col_name = f"{class_name}_{metric}"
                if col_name not in self.results:
                    self.results[col_name] = []
                self.results[col_name].append(val_metrics['per_class'][metric][i])

    def save_results(self):
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Create Excel writer
        with pd.ExcelWriter(self.output_file, engine='openpyxl') as writer:
            # Write overview sheet
            overview_cols = ['timestamp', 'model_name', 'epoch', 'train_loss', 'val_loss',
                           'train_macro_recall', 'train_macro_precision', 'train_macro_f1', 'train_macro_auc',
                           'val_macro_recall', 'val_macro_precision', 'val_macro_f1', 'val_macro_auc']
            df[overview_cols].to_excel(writer, sheet_name='Overview', index=False)
            
            # Write per-class metrics sheet
            class_metrics = [col for col in df.columns if any(metric in col for metric in ['recall', 'precision', 'f1', 'auc'])]
            df[['timestamp', 'model_name', 'epoch'] + class_metrics].to_excel(writer, sheet_name='Per-Class-Metrics', index=False)
            
            # Write best results sheet
            best_results = df.loc[df.groupby('model_name')['val_loss'].idxmin()]
            best_results.to_excel(writer, sheet_name='Best-Results', index=False)