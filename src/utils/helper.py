# src/utils/helpers.py

import os
import requests
from tqdm import tqdm
import hashlib

def setup_image_storage():
    base_dir = 'data/images'
    os.makedirs(base_dir, exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(base_dir, split), exist_ok=True)
    
    return base_dir

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
