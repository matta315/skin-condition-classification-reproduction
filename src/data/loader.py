# src/data/loader.py

import pandas as pd
import os
from tqdm import tqdm
import requests
from sklearn.model_selection import train_test_split
import shutil
from ..utils.helpers import download_image, setup_image_storage

class DataLoader:
    def __init__(self, csv_path='data/fitzpatrick17k.csv'):
        """
        Initialize DataLoader with path to CSV file
        Args:
            csv_path: Path to the Fitzpatrick17k dataset CSV
        """
        self.csv_path = csv_path
        self.base_dir = setup_image_storage()
        
    def download_dataset_images(self, df):
        """
        Download images from URLs using md5hash as filename
        Args:
            df: DataFrame containing image URLs and md5hashes
        Returns:
            list of successful download paths
        """
        print("Downloading images...")
        successful_downloads = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_filename = f"{row['md5hash']}.jpg"
            filepath = os.path.join(self.base_dir, image_filename)
            
            if os.path.exists(filepath):
                successful_downloads.append(filepath)
                continue
                
            if download_image(row['url'], filepath):
                successful_downloads.append(filepath)
        
        return successful_downloads
    
    def update_df_with_image_paths(self, df):
        """
        Add local image paths to DataFrame and verify existence
        Args:
            df: DataFrame to update
        Returns:
            DataFrame with verified image paths
        """
        df['image_path'] = df['md5hash'].apply(
            lambda x: os.path.join(self.base_dir, f"{x}.jpg")
        )
        
        df['image_exists'] = df['image_path'].apply(os.path.exists)
        df_with_images = df[df['image_exists']].copy()
        df_with_images.drop('image_exists', axis=1, inplace=True)
        
        return df_with_images
    
    def create_data_splits(self, df_with_images, train_size=0.7, val_size=0.15):
        """
        Split data and organize images into train/val/test folders
        Args:
            df_with_images: DataFrame with image paths
            train_size: Proportion for training set
            val_size: Proportion for validation set
        Returns:
            train_df, val_df, test_df: Split DataFrames
        """
        print("Creating data splits...")
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = os.path.join(self.base_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            print(f"Created directory: {split_dir}")

        # Create splits
        try:
            train_df, temp_df = train_test_split(
                df_with_images,
                train_size=train_size,
                stratify=df_with_images['fitzpatrick_scale'],
                random_state=42
            )
            
            val_size_adjusted = val_size / (1 - train_size)
            val_df, test_df = train_test_split(
                temp_df,
                train_size=val_size_adjusted,
                stratify=temp_df['fitzpatrick_scale'],
                random_state=42
            )
        except Exception as e:
            print(f"Error during splitting: {str(e)}")
            return None, None, None

        # Move images to split folders
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_dir = os.path.join(self.base_dir, split_name)
            print(f"\nMoving images to {split_name} folder...")
            
            new_paths = []
            for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
                src_path = row['image_path']
                dst_path = os.path.join(split_dir, f"{row['md5hash']}.jpg")
                
                try:
                    if os.path.exists(src_path):
                        if not os.path.exists(dst_path):
                            shutil.copy2(src_path, dst_path)
                        new_paths.append(dst_path)
                    else:
                        new_paths.append(None)
                except Exception as e:
                    print(f"Error moving {row['md5hash']}: {str(e)}")
                    new_paths.append(None)
            
            split_df['image_path'] = new_paths
            split_df = split_df.dropna(subset=['image_path'])

        # Save splits to CSV
        for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            split_df.to_csv(os.path.join(self.base_dir, f'{split_name}_split.csv'), index=False)

        return train_df, val_df, test_df
    
    def cleanup_main_folder(self, train_df, val_df, test_df):
        """
        Remove images from main folder that have been moved to splits
        """
        print("\nCleaning up main images folder...")
        
        split_images = set()
        for df in [train_df, val_df, test_df]:
            split_images.update(df['md5hash'].apply(lambda x: f"{x}.jpg"))
        
        main_images = [f for f in os.listdir(self.base_dir) 
                      if f.endswith('.jpg') and os.path.isfile(os.path.join(self.base_dir, f))]
        
        removed_count = 0
        for image in main_images:
            if image in split_images:
                try:
                    os.remove(os.path.join(self.base_dir, image))
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {image}: {str(e)}")
        
        print(f"Removed {removed_count} duplicate images from main folder")
    
    def prepare_dataset(self):
        """
        Complete pipeline: load data, download images, create splits, and cleanup
        Returns:
            train_df, val_df, test_df: Split DataFrames
        """
        # Load CSV
        print(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Download images
        successful_downloads = self.download_dataset_images(df)
        print(f"Successfully downloaded {len(successful_downloads)} images")
        
        # Update DataFrame with image paths
        df_with_images = self.update_df_with_image_paths(df)
        print(f"Total samples with images: {len(df_with_images)}")
        
        # Create splits
        train_df, val_df, test_df = self.create_data_splits(df_with_images)
        
        # Cleanup main folder
        self.cleanup_main_folder(train_df, val_df, test_df)
        
        return train_df, val_df, test_df

# Run the split data preparation
if __name__ == "__main__":
    loader = DataLoader()
    train_df, val_df, test_df = loader.prepare_dataset()
