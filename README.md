# eNeuralFAM

**eNeuralFAM** is a hybrid classification framework that combines a shallow Artificial Neural Network (ANN) for feature extraction with a Fuzzy ARTMAP (FAM) classifier for online learning. The model is designed to handle noisy data streams and performs incremental training across folds.

## Description

This project implements an ensemble-based online learning model using:
- ANN for feature extraction
- Fuzzy ARTMAP (FAM) as the classifier
- Ensemble voting mechanism for robust predictions

The model is evaluated across multiple datasets and noise levels, with performance tracked using various classification metrics.

## Datasets

The datasets used in this project can be downloaded from the [KEEL repository](http://keel.es/). 

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- TensorFlow or Keras
