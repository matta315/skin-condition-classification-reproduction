import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from .dataset import SkinConditionDataset
from .preprocess import get_data_transforms

def load_data(csv_dir, batch_size=32, num_workers=4):
    """
    Load the preprocessed data and create DataLoaders.
    
    Args:
        csv_dir (str): Directory containing the CSV files
        batch_size (int): Batch size for DataLoaders
        num_workers (int): Number of workers for DataLoaders
    
    Returns:
        dict: Dictionary containing train, validation, and test DataLoaders
    """
    # Load the split datasets
    train_df = pd.read_csv(os.path.join(csv_dir, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(csv_dir, 'val_split.csv'))
    test_df = pd.read_csv(os.path.join(csv_dir, 'test_split.csv'))
    
    # Get data transforms
    transforms = get_data_transforms()
    
    # Create datasets
    train_dataset = SkinConditionDataset(train_df, transform=transforms['train'])
    val_dataset = SkinConditionDataset(val_df, transform=transforms['eval'])
    test_dataset = SkinConditionDataset(test_df, transform=transforms['eval'])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': len(train_df['label_encoded'].unique())
    }