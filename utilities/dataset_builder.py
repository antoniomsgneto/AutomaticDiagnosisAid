import csv
import os

CSV_FILE = "/home/antonio/PycharmProjects/AutomaticDiagnosisAid/dataset_info/info.csv"
header = ['Patient_Designation', 'Patient_Number']


class DataStorage:
    def __init__(self):
        file = open(CSV_FILE, 'w')
        self.writer = csv.writer(file)
        self.writer.writerow(header)
        self.patient_counter = 0

    def build_dataset(self, path_to_main_folder):
        for file in os.listdir(path_to_main_folder):
            body = [file, self.patient_counter]
            self.writer.writerow(body)
            self.patient_counter += 1


def main():
    DataStorage().build_dataset("/home/antonio/Downloads/Tese")


if __name__ == "__main__":
    main()
