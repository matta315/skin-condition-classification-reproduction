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


# In src/data/dataset.py
class FitzpatrickDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        
        # Get all image paths and labels
        self.samples = []
        for condition in ['benign', 'malignant', 'non_neoplastic']:
            condition_dir = os.path.join(self.data_dir, condition)
            for skin_type in range(1, 7):
                type_dir = os.path.join(condition_dir, f'type_{skin_type}')
                if os.path.exists(type_dir):
                    for img_name in os.listdir(type_dir):
                        self.samples.append({
                            'path': os.path.join(type_dir, img_name),
                            'condition': condition,
                            'skin_type': skin_type
                        })