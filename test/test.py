# main.py

from src.prepare_dataset import prepare_dataset

if __name__ == "__main__":
    train_df, val_df, test_df = prepare_dataset()
    
    # Print some statistics
    print("\nDataset splits:")
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
