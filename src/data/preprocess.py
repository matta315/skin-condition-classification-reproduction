
# to download images from URLs in a CSV file and organize them into directories based on conditions and skin types.

import os
import pandas as pd
import requests
from tqdm import tqdm
import shutil
from sklearn.model_selection import train_test_split

# Helper function for downloading images using request from terminal
def download_image(url, filepath):
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

# Download function
def download_dataset_images(df, base_dir):
    """
    Download images from URLs and save them using md5hash as filename
    """
    print("Downloading images...")
    successful_downloads = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Create filepath using md5hash
        image_filename = f"{row['md5hash']}.jpg"
        filepath = os.path.join(base_dir, image_filename)
        
        # Skip if image already exists
        if os.path.exists(filepath):
            successful_downloads.append(filepath)
            continue
            
        # Download image if it doesn't exist
        if download_image(row['url'], filepath):
            successful_downloads.append(filepath)
    
    return successful_downloads

# Function to update DataFrame with image paths
def update_df_with_image_paths(df, base_dir):
    """
    Add local image paths to DataFrame and verify existence
    Args:
        df: DataFrame to update
    Returns:
        DataFrame with verified image paths
    """
    df['image_path'] = df['md5hash'].apply(
        lambda x: os.path.join(base_dir, f"{x}.jpg")
    )
    
    df['image_exists'] = df['image_path'].apply(os.path.exists)
    df_with_images = df[df['image_exists']].copy()
    df_with_images.drop('image_exists', axis=1, inplace=True)
    
    return df_with_images

# # Organize function
# def organize_images(csv_file, output_dir):
#     """
#     Organize images by condition and skin type
#     """
#     df = pd.read_csv(csv_file)
    
#     for _, row in df.iterrows():
#         # Get paths
#         condition = row['three_partition_label']
#         skin_type = f"type_{row['fitzpatrick_scale']}"
        
#         # Create directories if they don't exist
#         save_dir = os.path.join(output_dir, condition, skin_type)
#         os.makedirs(save_dir, exist_ok=True)
        
#         # Download and save image
#         image_url = row['url']
#         image_path = os.path.join(save_dir, f"{row['md5hash']}.jpg")
#         if not os.path.exists(image_path):
#             download_image(image_url, image_path)

def create_data_splits(df_with_images,base_dir='data/processed/images',train_size=0.7, val_size=0.15):
    """
    Split data and organize images into train/val/test folders with hierarchical structure
    Args:
        df_with_images: DataFrame with image paths
        train_size: Proportion for training set
        val_size: Proportion for validation set
    Returns:
        train_df, val_df, test_df: Split DataFrames
    """
    print("Creating data splits...")
    
    # Create directories if they don't exist
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(base_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        print(f"Created directory: {split_dir}")
        
        # Create classification directories within each split
        for classification in ['benign', 'malignant', 'non_neoplastic']:
            class_dir = os.path.join(split_dir, classification)
            os.makedirs(class_dir, exist_ok=True)
    print ('Done Creating directories')
    # Create splits
    try:
        # Stratify by both fitzpatrick_scale and three_partition_label if possible
        # To help identify classification: benign, non-neoplastic, malignant
        if 'three_partition_label' in df_with_images.columns:
            # Create a combined stratification column
            df_with_images['strat_col'] = df_with_images['fitzpatrick_scale'].astype(str) + '_' + df_with_images['three_partition_label']
            strat_col = 'strat_col'
        else:
            strat_col = 'fitzpatrick_scale'

        train_df, temp_df = train_test_split(
            df_with_images,
            train_size=train_size,
            stratify=df_with_images[strat_col],
            random_state=42
        )
        
        val_size_adjusted = val_size / (1 - train_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size_adjusted,
            stratify=temp_df[strat_col],
            random_state=42
        )
        
        print("\nInitial split completed:")
        print(f"Training set: {len(train_df)} images")
        print(f"Validation set: {len(val_df)} images")
        print(f"Test set: {len(test_df)} images")

        # Return here if you just want to test the splitting
        return train_df, val_df, test_df
        
    except Exception as e:
        print(f"Error during splitting: {str(e)}")
        return None, None, None

    # # Move images to hierarchical folders
    # for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    #     split_dir = os.path.join(base_dir, split_name)
    #     print(f"\nOrganizing images in {split_name} folder...")
        
    #     new_paths = []
    #     for _, row in tqdm(split_df.iterrows(), total=len(split_df)):
    #         try:
    #             src_path = row['image_path']
                
    #             # Get classification category (benign/malignant/non_neoplastic)
    #             if 'three_partition_label' in row:
    #                 classification = row['three_partition_label']
    #             else:
    #                 # Default to non_neoplastic if not specified
    #                 classification = 'non_neoplastic'
                
    #             # Get condition (label) and skin type
    #             condition = row['label'].replace('/', '_').replace(' ', '_')
    #             skin_type = f"type_{row['fitzpatrick_scale']}"
                
    #             # Create condition directory if it doesn't exist
    #             condition_dir = os.path.join(split_dir, classification, condition)
    #             os.makedirs(condition_dir, exist_ok=True)
                
    #             # Create skin type directory if it doesn't exist
    #             skin_type_dir = os.path.join(condition_dir, skin_type)
    #             os.makedirs(skin_type_dir, exist_ok=True)
                
    #             # Define destination path
    #             dst_path = os.path.join(skin_type_dir, f"{row['md5hash']}.jpg")
                
    #             # Copy the file if it exists
    #             if os.path.exists(src_path):
    #                 if not os.path.exists(dst_path):
    #                     shutil.copy2(src_path, dst_path)
    #                 new_paths.append(dst_path)
    #             else:
    #                 print(f"Warning: Source image not found: {src_path}")
    #                 new_paths.append(None)
    #         except Exception as e:
    #             print(f"Error organizing image {row.get('md5hash', 'unknown')}: {str(e)}")
    #             new_paths.append(None)
        
    #     # Update image paths in DataFrame
    #     split_df['image_path'] = new_paths
    #     split_df = split_df.dropna(subset=['image_path'])

    # # Save splits to CSV
    # for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    #     split_df.to_csv(os.path.join(self.base_dir, f'{split_name}_split.csv'), index=False)
    #     print(f"Saved {split_name}_split.csv with {len(split_df)} records")

    # # Print final directory structure summary
    # print("\nFinal directory structure created:")
    # for split in ['train', 'val', 'test']:
    #     split_dir = os.path.join(self.base_dir, split)
    #     classifications = os.listdir(split_dir)
    #     for classification in classifications:
    #         class_dir = os.path.join(split_dir, classification)
    #         if os.path.isdir(class_dir):
    #             conditions = os.listdir(class_dir)
    #             print(f"  {split}/{classification}: {len(conditions)} conditions")

    # return train_df, val_df, test_df

# Main function to test the code
def main():
    # Set paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    
    # Set paths relative to project root
    csv_path = os.path.join(project_root, 'data', 'raw', 'fitzpatrick17k.csv')
    base_dir = os.path.join(project_root, 'data', 'processed', 'images')
    
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Step 1: Load the CSV file
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    #df = df.iloc[0:5]
    df = df[df['label'] == 'psoriasis']
    print(f"Loaded {len(df)} records from CSV")
    print(df)
    
    # # Step 2: Download images (optional - can be skipped if already downloaded)
    # download_option = input("Download images? (y/n): ").lower()
    # if download_option == 'y':
    #     successful_downloads = download_dataset_images(df, base_dir)
    #     print(f"Successfully downloaded {len(successful_downloads)} images")
    
    # Step 3: Update DataFrame with image paths
    df_with_images = update_df_with_image_paths(df, base_dir)
    print(f"Total samples with images: {len(df_with_images)}")
    
    # Step 4: Create data splits and organize images
    split_option = input("Create data splits and organize images? (y/n): ").lower()
    if split_option == 'y':
        train_df, val_df, test_df = create_data_splits(df_with_images, base_dir)
        
        # Print summary
        if train_df is not None:
            print("\nData preparation completed successfully!")
            print(f"Training set: {len(train_df)} images")
            print(f"Validation set: {len(val_df)} images")
            print(f"Test set: {len(test_df)} images")
            
            # Print distribution of skin types in each split
            print("\nFitzpatrick scale distribution:")
            print("Training set:")
            print(train_df['fitzpatrick_scale'].value_counts().sort_index())
            print("\nValidation set:")
            print(val_df['fitzpatrick_scale'].value_counts().sort_index())
            print("\nTest set:")
            print(test_df['fitzpatrick_scale'].value_counts().sort_index())
            
            # Print distribution of classifications in each split
            if 'three_partition_label' in train_df.columns:
                print("\nClassification distribution:")
                print("Training set:")
                print(train_df['three_partition_label'].value_counts())
                print("\nValidation set:")
                print(val_df['three_partition_label'].value_counts())
                print("\nTest set:")
                print(test_df['three_partition_label'].value_counts())

if __name__ == "__main__":
    main()