# Model Selection and SVM for Blockchain Node Analysis

This repository contains Python scripts for performing model selection and Support Vector Machine (SVM) analysis on blockchain node data. The goal is to classify blockchain nodes and identify potentially malicious ones based on various features.

## Contents

1. [Model Selection](#model-selection)
2. [SVM Analysis](#svm-analysis)

## Model Selection

File: `model_selection.py`

### Description
The `model_selection.py` script performs model selection using Random Forest, Support Vector Machine, and Multi-Layer Perceptron classifiers on blockchain node data.

### Steps
1. Data loading and preprocessing.
2. Splitting the dataset into training and testing sets.
3. Constructing pipelines for each classifier (Random Forest, SVM, MLP).
4. Performing grid search with cross-validation to find the best hyperparameters for each classifier.
5. Evaluating the models using accuracy score and classification reports.
6. Selecting the best-performing model based on accuracy.

## SVM Analysis

File: `SVM.py`

### Description
The `SVM.py` script implements SVM analysis specifically for identifying potentially malicious blockchain nodes.

### Steps
1. Data loading.
2. Data preprocessing, including feature selection and scaling.
3. Training an SVM model with linear kernel.
4. Making predictions on the dataset.
5. Identifying potentially malicious nodes based on the predictions.

## Usage

To run the scripts:

```bash
python model_selection.py
python SVM.py
```

Ensure that the necessary dependencies are installed, including `pandas`, `scikit-learn`, and others specified in `requirements.txt`.

## Data

The scripts expect the blockchain node data to be provided in a CSV file named `Blockchain103.csv`.

## Dependencies

- pandas
- scikit-learn
