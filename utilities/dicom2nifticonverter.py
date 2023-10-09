
import sys
from nibabel import Nifti1Image
import pydicom as dicom
from shutil import copyfile
import numpy as np
import nibabel as nib
import os
import gzip
import shutil

SHORT_AXIS_FOLDER_NAME = ['/SACine401/', '/SAcine401/', '/sacine401/','2CHCine401/','/saCine401/']


def full_convert(path_to_dicom_folder, path_to_result_folder, path_to_nifti_folder, patient_identification):
    __make_folder(path_to_result_folder)
    __make_folder(path_to_nifti_folder)
    # Get main folder and divide images between Cuts and inside Cuts between Slices
    convert_to_folders(path_to_dicom_folder, path_to_result_folder)
    folder = max(SHORT_AXIS_FOLDER_NAME, key=lambda f: count_directory_size(path_to_result_folder + f))
    path_to_short_axis = path_to_result_folder + folder
    count = 0

    if os.path.exists(path_to_short_axis):
        # Sort short axis slice folders from lowest to highest
        file_list = os.listdir(path_to_short_axis)
        file_list.sort(key=lambda x: float(x))
        for folder in file_list:
            print("Converting slice: " + folder)
            # Get all files from slice folder convert to numpy array and from that to nifti
            convert_slice_folder_to_nifti(path_to_short_axis + '/' + folder,
                                          path_to_nifti_folder + '/' + patient_identification + '_' + str(count) + '_0000')
            count += 1
    gzip_files(path_to_nifti_folder)
    shutil.rmtree(path_to_result_folder)


def count_directory_size(dir_path: str) -> int:
    count = 0
    # Iterate directory
    if os.path.exists(dir_path):
        for path in os.listdir(dir_path):
            # check if current path is a file
            if os.path.isdir(os.path.join(dir_path, path)):
                count += 1
    return count


def convert_to_folders(path_to_dicom_folder, path_to_result_folder):
    list_of_folders = []
    for file in os.listdir(path_to_dicom_folder):
        path_to_file = path_to_dicom_folder + '/' + file
        ds = dicom.read_file(path_to_file)
        try:
            path_to_new_folder = path_to_result_folder + '/' + str(ds[0x0008103e].value) + '/' + str(
                ds[0x00201041].value)
            path_to_new_folder = path_to_new_folder.replace("'", "")
            path_to_new_folder = path_to_new_folder.replace(" ", "")
            print("Processing file:" + file + "to folder")
            if not os.path.exists(path_to_new_folder):
                os.makedirs(path_to_new_folder)
                list_of_folders.append(path_to_new_folder)
            copyfile(path_to_file, path_to_new_folder + '/' + str(file))
        except KeyError:
            print('Unidentified series description or location - Image discarded')
    return list_of_folders


def __get_trigger_time(ds) -> int:
    return dicom.read_file(ds)[0x00181060].value


def  convert_slice_folder_to_nifti(path_to_dicom_folder, path_to_result_folder):
    unstacked_list = []
    width = 512
    height = 512
    # DEFAULT VALUES CHANGE FOR LATEST FOUND
    # Sort list
    file_list = os.listdir(path_to_dicom_folder)
    file_list.sort(key=lambda x: __get_trigger_time(path_to_dicom_folder + '/' + x))
    first_file = file_list[0]
    ds = dicom.read_file(path_to_dicom_folder + '/' + first_file)
    print('Pixel Spacing:', ds[0x0028, 0x0030].value)
    print('Slice Thickness:', ds[0x0018, 0x0050].value)

    for file in file_list:
        print("Processing" + file + "to Nifti")
        path_to_file = path_to_dicom_folder + '/' + file
        ds = dicom.read_file(path_to_file)
        width = len(ds.pixel_array)
        height = len(ds.pixel_array[0])
        unstacked_list.append(ds.pixel_array)
    result = convert_list_of_pixel_array_to_nifti(unstacked_list, width, height)
    result_header = result.header.copy()
    pixdim = result_header['pixdim']
    pixdim[1] = ds[0x0028, 0x0030].value[0]
    pixdim[2] = ds[0x0028, 0x0030].value[1]
    pixdim[3] = ds[0x0018, 0x0050].value
    result_header['pixdim'] = pixdim
    result = result.__class__(result.get_fdata(), result.affine,result_header)
    nib.save(result, path_to_result_folder)

def convert_list_of_pixel_array_to_nifti(list_of_arrays, width, height) -> Nifti1Image:
    nr_of_slices = len(list_of_arrays)
    result_list = np.empty(shape=(width, height, nr_of_slices), dtype=np.float32)
    for i, array in enumerate(list_of_arrays):
        result_list[:, :, i] = array
    return nib.Nifti1Image(result_list, affine=np.eye(4))


def gzip_files(folder):
    print("Zipping files...")
    for file in os.listdir(folder):
        path_to_file = folder + '/' + file
        if file.endswith('.nii'):
            with open(path_to_file, 'rb') as src, gzip.open(path_to_file + '.gz', 'wb') as dst:
                dst.writelines(src)
            os.remove(path_to_file)


def __make_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_array_list_to_nifti(list_of_arrays, width, height,path_to_result_folder, affine,header):
    print("Saving nifti")
    result = convert_list_of_pixel_array_to_nifti(list_of_arrays, width, height)
    result = result.__class__(result.get_fdata(), affine, header)
    nib.save(result, path_to_result_folder)

def main():
    if sys.argv[0]:
        for count, dir in enumerate(os.listdir(sys.argv[1])):
            patient_identification = dir
            full_convert(sys.argv[1] + '/' + dir, sys.argv[2], sys.argv[3], patient_identification)


if __name__ == "__main__":
    main()
