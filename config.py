# I
DATA_PATH = "data/processed"
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.001

MODEL_CONFIG = {
    'num_classes': 3,  # benign, malignant, non-neoplastic
    'dropout_rate': 0.4
}

# Directory structure
DIRS = {
    'data': {
        'raw': 'data/raw',
        'processed': 'data/processed',
        'interim': 'data/interim'
    },
    'models': 'models',
    'results': 'results'
}
