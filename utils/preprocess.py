#preprocess.py

import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

def prepare_data(csv_path, img_dir, train_list_path, test_list_path):
   with open(train_list_path) as f:
       train_images = set(x.strip() for x in f.readlines())
   with open(test_list_path) as f:
       test_images = set(x.strip() for x in f.readlines())
   
   df = pd.read_csv(csv_path)
   existing_images = set(os.listdir(img_dir))
   
   train_df = df[df['Image Index'].isin(train_images & existing_images)]
   test_df = df[df['Image Index'].isin(test_images & existing_images)]
   
   print(f"Training images: {len(train_df)}")
   print(f"Test images: {len(test_df)}")
   
   return train_df, test_df

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