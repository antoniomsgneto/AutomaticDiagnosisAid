from os import listdir
#from segmentation.simple_segmentation import segment_patient
from segmentation.dot_segmentation import segment_patient
import sys
import pickle5 

if __name__ == '__main__':
    if sys.argv[1]:
        counter = 0
        data_dict={}
        for file in listdir(sys.argv[1]):
            if file.endswith('nii.gz'):
                print("here")
                data_by_frame= segment_patient(sys.argv[2] + file.replace('.nii.gz','_0000.nii.gz'), sys.argv[1] + file, sys.argv[3])
                try:
                    data_dict[counter].append(data_by_frame)
                except KeyError:
                    data_dict[counter] = data_by_frame
                counter+=1
        pickle5.dump( data_dict, open( "save_data.p", "wb" ) )

