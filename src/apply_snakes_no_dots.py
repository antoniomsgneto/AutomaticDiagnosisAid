from os import listdir
from segmentation.segmentation_with_sampling import segment_patient
import sys

if __name__ == '__main__':
    if sys.argv[1]:
        for file in listdir(sys.argv[1]):
            if file.endswith('nii.gz'):
                print("here")
                segment_patient(sys.argv[1], sys.argv[2])