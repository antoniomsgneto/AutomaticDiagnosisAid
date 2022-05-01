from os import listdir
from segmentation.measure import Measures
import sys

if __name__ == '__main__':
    if sys.argv[1]:
        patient_dict = {}
        for file in sorted(listdir(sys.argv[1])):
            if file.endswith('nii.gz'):
                patient = file.split('_')[0]
                if patient in patient_dict.keys():
                    patient_dict[patient].append(sys.argv[1]+'/'+file)
                else:
                    patient_dict[patient] = [sys.argv[1]+'/'+file]

        for patient in patient_dict.keys():
            measure: Measures = Measures()
            print(patient)
            print("--------------------------------------------")
            ef, ed, es = measure.full_volume_and_ef_calculate(patient_dict[patient])
            es2, ed2 = measure.outer_volume(patient_dict[patient])
            if sys.argv[2]:
                file = open(sys.argv[2], "a")
                file.writelines(f'{patient} : Ejection Fraction = {ef} % End Diastolic Volume = {ed} ml' +
                           f' End Sistolic Volume = {es} ml \n End Diastolic Outer Volume = {ed2} ' +
                           f'\n End Sistolic Outer Volume = {es2}')
                file.writelines('\n')
                file.close()
