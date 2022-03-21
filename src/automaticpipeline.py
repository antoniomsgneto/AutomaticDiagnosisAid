
import os
import sys


def main():
    from utilities import dicom2nifticonverter as converter
    if sys.argv[0]:
        for count, directory in enumerate(os.listdir(sys.argv[1])):
            patient_identification = '/Patient10' + str(count) + '_'
            converter.full_convert(sys.argv[1] + '/' + directory, sys.argv[2], sys.argv[3], patient_identification)
    os.system('nnUnet_predict -i ' + sys.argv[3] + ' -o' + sys.argv[4] + '-t 27 -m 2d --save')


if __name__ == '__main__':
    main()
