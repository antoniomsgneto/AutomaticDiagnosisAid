#!/bin/bash

echo "Setting up directories"

export nnUNet_preprocessed = $PATH_TO_PREPROCESSED
export RESULTS_FOLDER = $PATH_TO_RESULTS
export nnUnet_raw_data = $PATH_TO_DATA

nnUNet_plan_and_preprocess -t 27

