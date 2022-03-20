#!/bin/bash

echo "Setting up directories"

export nnUNet_preprocessed = /home/amsgn/Dados/PreProcessed
export RESULTS_FOLDER = /home/amsgn/Dados/Results
export nnUnet_raw_data = /home/amsgn/Dados/Nifti

nnUNet_plan_and_preprocess -t 27

