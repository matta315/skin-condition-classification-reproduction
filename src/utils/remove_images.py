#!/usr/bin/env python3
"""
Utility script to remove all image files from the repository.
"""

import os
import re
import argparse
from pathlib import Path


def is_image_file(filename):
    """Check if a file is an image based on its extension."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    return any(filename.lower().endswith(ext) for ext in image_extensions)


def remove_images(directory, dry_run=True, excluded_dirs=None, include_data_folders=False):
    """
    Remove all image files from the specified directory and its subdirectories.
    
    Args:
        directory (str): The directory to scan for images
        dry_run (bool): If True, only print files that would be removed without actually removing them
        excluded_dirs (list): List of directory names to exclude from scanning
        include_data_folders (bool): If True, also remove images from train/val/test folders
    
    Returns:
        int: Number of files removed or that would be removed
    """
    if excluded_dirs is None:
        excluded_dirs = ['.git', 'node_modules', '__pycache__', 'venv', '.env']
    
    # By default, exclude data folders unless explicitly included
    data_folders = ['train', 'val', 'test', 'validation']
    if not include_data_folders:
        excluded_dirs.extend(data_folders)
    
    count = 0
    total_size = 0
    
    for root, dirs, files in os.walk(directory):
        # Check if this directory or any parent directory should be excluded
        path_parts = Path(root).parts
        skip_dir = False
        
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        # Additional check for data folders in path
        if not include_data_folders:
            if any(folder in path_parts for folder in data_folders):
                continue
        
        for file in files:
            if is_image_file(file):
                file_path = os.path.join(root, file)
                try:
                    size = os.path.getsize(file_path)
                    total_size += size
                    
                    if dry_run:
                        print(f"Would remove: {file_path} ({size / 1024:.2f} KB)")
                    else:
                        print(f"Removing: {file_path} ({size / 1024:.2f} KB)")
                        os.remove(file_path)
                    
                    count += 1
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
    
    print(f"\nSummary:")
    action = "Would remove" if dry_run else "Removed"
    print(f"{action} {count} image files")
    print(f"Total size: {total_size / (1024 * 1024):.2f} MB")
    
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove all image files from a directory")
    parser.add_argument("--directory", "-d", default=".", 
                        help="Directory to scan (default: current directory)")
    parser.add_argument("--execute", "-e", action="store_true", 
                        help="Actually remove files (without this flag, it runs in dry-run mode)")
    parser.add_argument("--exclude", "-x", nargs="+", default=[],
                        help="Additional directories to exclude")
    parser.add_argument("--include-data", "-i", action="store_true",
                        help="Include images in train/val/test folders (these are excluded by default)")
    
    args = parser.parse_args()
    
    directory = os.path.abspath(args.directory)
    excluded_dirs = ['.git', 'node_modules', '__pycache__', 'venv', '.env'] + args.exclude
    
    print(f"Scanning directory: {directory}")
    print(f"Excluded directories: {', '.join(excluded_dirs)}")
    if not args.include_data:
        print("Note: train/val/test folders are excluded by default. Use --include-data to include them.")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN (no files will be deleted)'}\n")
    
    remove_images(directory, not args.execute, excluded_dirs, args.include_data)