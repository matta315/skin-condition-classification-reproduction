import torch
from torch.utils.data import Dataset
from PIL import Image

class SkinConditionDataset(Dataset):
    """Dataset for skin condition classification"""
    
    def __init__(self, df, transform=None):
        """
        Args:
            df (pandas.DataFrame): DataFrame containing image paths and labels
            transform (callable, optional): Optional transform to be applied on images
        """
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Get labels
        label = row['label_encoded']
        fitzpatrick_scale = row['fitzpatrick_scale']
        
        return {
            'image': image,
            'label': label,
            'fitzpatrick_scale': fitzpatrick_scale
        }
