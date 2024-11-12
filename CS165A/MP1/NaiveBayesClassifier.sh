#!/bin/bash

# Paths to traing and validation paths
TRAIN_DATA_PATH="./train_data.csv"
VALIDATE_DATA_PATH="./validation_data.csv"

# Run the Python script with paths to the data files
python3 NaiveBayesClassifier.py "$TRAIN_DATA_PATH" "$VALIDATE_DATA_PATH"
