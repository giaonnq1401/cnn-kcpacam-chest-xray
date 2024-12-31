#preprocess.py

import pandas as pd
import os
import random 
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def prepare_data(csv_path, img_dir, train_list_path, test_list_path, train_size=None, test_size=None):
    """
    Prepare training and testing datasets with controlled sample sizes
    Args:
        csv_path (str): Path to CSV file containing image labels
        img_dir (str): Directory containing images
        train_list_path (str): Path to file containing training image names
        test_list_path (str): Path to file containing test image names
        train_size (int, optional): Number of samples for training set. If None, use all available
        test_size (int, optional): Number of samples for test set. If None, use all available
    Returns:
        tuple: (train_df, test_df) Pandas DataFrames containing selected samples
    """
    # Read image lists
    with open(train_list_path) as f:
        train_images = list(x.strip() for x in f.readlines())
    with open(test_list_path) as f:
        test_images = list(x.strip() for x in f.readlines())
    
    # Get existing images
    df = pd.read_csv(csv_path)
    existing_images = set(os.listdir(img_dir))
    
    # Filter available images
    train_available = [img for img in train_images if img in existing_images]
    test_available = [img for img in test_images if img in existing_images]
    
    # Take first N samples if size is specified
    if train_size is not None:
        train_size = min(train_size, len(train_available))
        train_selected = train_available[:train_size]
    else:
        train_selected = train_available
        
    if test_size is not None:
        test_size = min(test_size, len(test_available))
        test_selected = test_available[:test_size]
    else:
        test_selected = test_available
    
    # Create dataframes
    train_df = df[df['Image Index'].isin(train_selected)]
    test_df = df[df['Image Index'].isin(test_selected)]
    
    print(f"Training images selected: {len(train_df)}")
    print(f"Test images selected: {len(test_df)}")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    for class_name in ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                      'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                      'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']:
        count = sum(train_df['Finding Labels'].str.contains(class_name))
        print(f"{class_name}: {count} ({count/len(train_df)*100:.2f}%)")
    
    return train_df, test_df

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    return train_transform, test_transform

class ChestXrayDataset(Dataset):
   def __init__(self, df, img_dir, transform=None):
       self.df = df
       self.img_dir = img_dir
       self.transform = transform
       self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 
                      'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 
                      'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
       
   def __len__(self):
       return len(self.df)
   
   def __getitem__(self, idx):
       img_path = os.path.join(self.img_dir, self.df.iloc[idx]['Image Index'])
       image = Image.open(img_path).convert('RGB')
       
       if self.transform:
           image = self.transform(image)
           
       finding_labels = [1 if c in self.df.iloc[idx]['Finding Labels'] else 0 
                        for c in self.classes]
       return image, torch.FloatTensor(finding_labels)