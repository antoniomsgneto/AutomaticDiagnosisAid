#!/usr/bin/env bash

set -e

echo "here"

export nnUNet_raw_data_base='/home/sharedfolder/LeftHeart/ACDC/RAW_DATABASE/'
export nnUNet_preprocessed='/home/sharedfolder/LeftHeart/ACDC/PREPROCESSED/'
export RESULTS_FOLDER='/home/sharedfolder/LeftHeart/ACDC/RESULTS_FOLDER/'

source /home/amsgn/venv/env/bin/activate

python -m src.convert_images /home/amsgn/Dados/Data /home/amsgn/Dados/Results /home/amsgn/Dados/Nifti

nnUNet_predict -i /home/amsgn/testing2 -o /home/amsgn/OUTPUT_TESTING -t 27 -m 2d --save

python -m src.apply_snakes /home/amsgn/OUTPUT_DIRECTORY_3D/ /home/amsgn/Dados/Nifti/  /home/amsgn/SnakeOutput

#python -m src.apply_snakes_no_dots /home/amsgn/OUTPUT_ACDC/ /home/amsgn/SnakeOutput

#python -m src.calculate_volumes /home/amsgn/OUTPUT_DIRECTORY_3D/ /home/amsgn/result_report.txt

deactivate

echo "done"