import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.classes = self._get_classes()

    def __len__(self):
        return len(self.dataframe)

    def _get_classes(self):
        # Extract unique classes from labels
        labels = []
        for item in self.dataframe['Finding Labels']:
            labels.extend(item.split('|'))
        return list(set(labels))

    def create_label_array(self, label_string):
        label_array = np.zeros(len(self.classes))
        for label in label_string.split('|'):
            label_array[self.classes.index(label)] = 1
        return label_array

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.dataframe.iloc[idx]['Image Index'])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        labels = self.create_label_array(self.dataframe.iloc[idx]['Finding Labels'])
        return image, labels

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms

def prepare_data(csv_path, img_dir, test_size=0.2, random_state=42):
    """
    Prepare train and validation dataframes
    Args:
        csv_path: path to Data_Entry_2017_v2020.csv
        img_dir: directory containing the images
        test_size: ratio for validation split
        random_state: random seed for reproducibility
    Returns:
        train_df, val_df: dataframes containing only existing images
    """
    # Read file CSV
    df = pd.read_csv(csv_path)
    
    # Check out what photos exist
    exists_mask = df['Image Index'].apply(lambda x: os.path.exists(os.path.join(img_dir, x)))
    
    # Filter and keep only rows where images exist
    df_filtered = df[exists_mask].copy()
    
    # Print information about the number of photos
    print(f"Total images in CSV: {len(df)}")
    print(f"Images found in directory: {sum(exists_mask)}")
    print(f"Missing images: {len(df) - sum(exists_mask)}")
    
    # Split data
    train_df, val_df = train_test_split(df_filtered, test_size=test_size, random_state=random_state)
    
    return train_df, val_df

def get_class_weights(dataframe):
    """
    Calculate class weights to handle imbalanced data
    """
    labels = []
    for item in dataframe['Finding Labels']:
        labels.extend(item.split('|'))
    
    class_counts = pd.Series(labels).value_counts()
    total_samples = len(labels)
    
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    return class_weights