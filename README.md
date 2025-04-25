# skin-condition-classification-reproduction
A reproduction study examining AI bias in dermatology classification across different skin types using the Fitzpatrick 17k Dataset. Implements VGG-16 based model to evaluate how deep learning performance varies across skin colors, reproducing key findings from Groh et al.'s 2021 paper.

## Project Overview
This reproduction study focuses on evaluating how deep learning models perform in classifying skin conditions across different skin types using the Fitzpatrick 17k Dataset. Our goal is to reproduce the paper's key findings about AI bias in dermatology classification across different skin types.

### Key Components
- Implementation of VGG-16 based classification model
- Analysis across different Fitzpatrick skin types
- Evaluation of model performance on various skin conditions
- Reproduction of key metrics and results

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
├── data/                  # Data processing and loading scripts
├── models/               # Model architecture and training
├── notebooks/           # Analysis notebooks
├── results/             # Experimental results and visualizations
├── src/                 # Source code
├── requirements.txt     # Project dependencies
└── README.md           # Project documentation```


