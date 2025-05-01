# skin-condition-classification-reproduction
A reproduction study examining AI bias in dermatology classification across different skin types using the Fitzpatrick 17k Dataset. Implements VGG-16 based model to evaluate how deep learning performance varies across skin colors, reproducing key findings from Groh et al.'s 2021 paper.

## Project Overview
This reproduction study focuses on evaluating how deep learning models perform in classifying skin conditions across different skin types using the Fitzpatrick 17k Dataset. Our goal is to reproduce the paper's key findings about AI bias in dermatology classification across different skin types.

There are going to be 4 stages of the project
Stage 1:
Stage 2:
Stage 3:
Stage 4:

### Key Components
- Implementation of VGG-16 based classification model
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

## Repository Structure
```skin-condition-classification-reproduction/
skin-condition-classification-reproduction/
├── lib/                              
│   ├── data/                         # Downloaded Fitzpatrick17k dataset
│   └── weights/                      # Saved model weights
│
├── notebooks/
│   ├── 1_data_preparation.ipynb      # Data loading and preprocessing
│   ├── 2_model_implementation.ipynb  # VGG16 model implementation
│   ├── 3_training_evaluation.ipynb   # Training and basic evaluation
│   └── 4_fitzpatrick_analysis.ipynb  # Skin type specific analysis
│
├── src/
│   ├── data/
│   │   ├── data_loader.py           # Data loading utilities
│   │   └── transforms.py            # Image transformations
│   │
│   ├── models/
│   │   ├── vgg16_model.py          # VGG16 model architecture
│   │   └── training.py             # Training functions
│   │
│   └── evaluation/
│       ├── metrics.py              # Evaluation metrics
│       └── fitzpatrick_analyzer.py # Skin type analysis
│
├── requirements.txt
└── README.md


## Requirements

To install requirements:

```setup
pip install -r requirements.txt

python3 -m venv venv

source venv/bin/activate && pip install panda
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Data Processing:
Under src/data there are 2 functions to help processing source data **fitzatrick17k.csv**
- Download full set of data
- Splitting data into 3 different folders: test, train and val

## Training

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

## Evaluation

To evaluate my model on ImageNet, run:

```eval
python eval.py --model-file mymodel.pth --benchmark imagenet
```

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models



## Results



## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
