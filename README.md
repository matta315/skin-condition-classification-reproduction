# skin-condition-classification-reproduction
This reproduction study investigates AI bias in dermatology classification across different skin types by implementing VGG-16 based models on the Fitzpatrick 17k Dataset containing 16,577 clinical images spanning 114 skin conditions. Our primary models include a binary classifier that distinguishes between malignant and non-malignant conditions with 90.4% accuracy, and a skin type analysis model that evaluates performance disparities between light and dark skin tones. Through this work, we aim to reproduce and validate key findings from Groh et al.'s 2021 paper, highlighting how deep learning performance varies across the Fitzpatrick skin type spectrum and contributing to more equitable AI applications in dermatology.

## Project Overview
This reproduction study focuses on evaluating how deep learning models perform in classifying skin conditions across different skin types using the Fitzpatrick 17k Dataset. Our goal is to reproduce the paper's key findings about AI bias in dermatology classification across different skin types.

There are going to be 5 stages of the project
Simple Model: Basic Classifier  
Stage 2: Skin Type analysis
Stage 3: Multi-class Classification
Stage 4: Compare Light and Dark Skin Performance
Stage 5: Add ITA Analysis

#### Hyperparameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 32
IMAGE_SIZE = 128
NUM_EPOCHS = 5
DROPOUT_RATE = 0.5
WEIGHT_DECAY = 1e-5
DATA_AUGMENTATION = True  # Whether to use data augmentation

### Key Components
- Implementation of similar model of VGG-16 based classification model
- Analysis across different Fitzpatrick skin types
- Evaluation of model performance on various skin conditions
- Reproduction of key metrics and results

** VGG-16 is a **convolutional neural network (CNN)** architecture that's widely used for image classification tasks

### Dataset
- Fitzpatrick 17k Dataset
- 16,577 clinical images
- 114 skin conditions
- Annotated with Fitzpatrick skin types

## Original Paper
- Title: "Evaluating Deep Neural Networks Trained on Clinical Images in Dermatology with the Fitzpatrick 17k Dataset"
- Authors: Matthew Groh, Caleb Harris, Luis Soenksen, et al.
- Paper URL: https://arxiv.org/abs/2104.09957
- Data URL: https://github.com/mattgroh/fitzpatrick17k/blob/main/fitzpatrick17k.csv

## Requirements

To install requirements:

```setup
pip install -r requirements.txt

python3 -m venv venv

source venv/bin/activate && pip install panda
```

## Data Processing:

The processing stored under ```src\preprocess.py```

Under src/data there are 2 functions to help processing source data **fitzatrick17k.csv**
- Download full set of data
- Splitting data into 3 different folders: test, train and val

**To Remove all image**

```
python src/utils/remove_images.py --directory /Users/mmnguyen/Documents/Matta_local_code/skin-condition-classification-reproduction --execute --include-data
```

## Training

### Model 1 - Classification
Base line implementation focusing on binary classification on skin condition

Model stored under jupyterNoteBook/basic_clasification_model.ipynb

to build the mode: 

```torch.save(model.state_dict(), 'basic_classification_model.pth')```

### Model 2 - SkinType Analysis

Closer reproduced model with original paper

Model stored under jupyterNoteBook/fitzpatrick_reproduce_model.ipynb

Focusing on 3 main skin disease labels:
- psoriasis
- squamous cell carcinoma
- lichen planus

To train the model(s) in the paper, run this command:

```torch.save(model.state_dict(), 'fitpatrick_reproduce_model.pth')
```


## Evaluation

Ech of the Jupyter Notebook you will find the session for Evaluation and result

