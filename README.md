# skin-condition-classification-reproduction
A reproduction study examining AI bias in dermatology classification across different skin types using the Fitzpatrick 17k Dataset. Implements VGG-16 based model to evaluate how deep learning performance varies across skin colors, reproducing key findings from Groh et al.'s 2021 paper.

## Project Overview
This reproduction study focuses on evaluating how deep learning models perform in classifying skin conditions across different skin types using the Fitzpatrick 17k Dataset. Our goal is to reproduce the paper's key findings about AI bias in dermatology classification across different skin types.

There are going to be 4 stages of the project
Stage 1: Basic Classifier  
Stage 2: Skin Type analysis
Stage 3: Multi-class Classification
Stage 4: Compare Light and Dark Skin Performance
Stage 5: Add ITA Analysis

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

### Model 2 - SkinType Analysis

Focusing on 3 main skin disease labels:
- psoriasis
- squamous cell carcinoma
- lichen planus

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
