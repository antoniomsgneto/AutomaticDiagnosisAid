from os import listdir
# from segmentation.simple_segmentation import segment_patient
from segmentation.full_segmentation import segment_patient
import sys

if __name__ == '__main__':
    if sys.argv[1]:
        for file in listdir(sys.argv[1]):
            if file.endswith('nii.gz'):
                print("here")
                segment_patient(sys.argv[1] + file, sys.argv[2])
