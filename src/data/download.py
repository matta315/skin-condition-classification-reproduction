import os
import requests
import pandas as pd
from tqdm import tqdm

def download_fitzpatrick17k_dataset(output_dir):
    """
    Download the Fitzpatrick 17k dataset CSV file.
    
    Args:
        output_dir (str): Directory to save the dataset
    
    Returns:
        str: Path to the downloaded CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # URL for the Fitzpatrick 17k dataset
    dataset_url = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
    
    # Path to save the CSV file
    csv_path = os.path.join(output_dir, "fitzpatrick17k.csv")
    
    # Check if file already exists
    if os.path.exists(csv_path):
        print(f"Dataset already exists at {csv_path}")
        return csv_path
    
    # Download the dataset
    print(f"Downloading Fitzpatrick 17k dataset to {csv_path}...")
    response = requests.get(dataset_url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with progress bar
        with open(csv_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        
        print(f"Dataset downloaded successfully to {csv_path}")
        
        # Validate the downloaded file
        try:
            df = pd.read_csv(csv_path)
            print(f"Dataset loaded successfully with {len(df)} rows")
        except Exception as e:
            print(f"Error validating dataset: {str(e)}")
            return None
        
        return csv_path
    else:
        print(f"Failed to download dataset: HTTP {response.status_code}")
        return None

if __name__ == "__main__":
    # Example usage
    download_fitzpatrick17k_dataset("data/raw")