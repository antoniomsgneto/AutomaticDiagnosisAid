#!/usr/bin/env bash

set -e

echo "here"

#export nnUNet_raw_data_base='/home/sharedfolder/LeftHeart/ACDC/RAW_DATABASE/'
#export nnUNet_preprocessed='/home/sharedfolder/LeftHeart/ACDC/PREPROCESSED/'
#export RESULTS_FOLDER='/home/sharedfolder/LeftHeart/ACDC/RESULTS_FOLDER/'


#source /home/amsgn/venv/env/bin/activate

#python -m src.convert_images /home/amsgn/Dados/Data /home/amsgn/Dados/Results /home/amsgn/Dados/Nifti #Converter Folder de Pacientes Dicom em Nifti (1 - Pasta com varios Pacientes Dicom) (2 - Folder aux) (3 - Pasta onde se guardam imagens convertidas)

#nnUNet_predict -i /home/amsgn/Dados/Nifti -o /home/amsgn/OUTPUT_DIRECTORY_3D -t 27 -m 2d --save #Comando para executar segmentação com nnUnet 

python3 -m src.apply_snakes /home/joao/Desktop/code_tese/Gota2/Gota2/teste/ /home/joao/Desktop/code_tese/Dados/  /home/joao/Desktop/code_tese/results/ #Comando para aplicar snakes com pontos (1 - Pasta com nifti segmentadas) (2-Pasta com Nifti originais) (3-Pasta para guardar resultados)

python3 -m src.calculate_volumes /home/joao/Desktop/code_tese/Gota2/Gota2/teste/ /home/joao/Desktop/code_tese/result_report.txt #Comando calcular volumes (1-Pasta nifti segmentadas) (2 - Txt resultados)

#python3 -m src.pickle_example /home/joao/Desktop/code_tese/AutomaticDiagnosisAid/save_data.p

#deactivate

echo "done"
