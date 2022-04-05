import nibabel as nib
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from scipy import ndimage


def calculate_values(img, slice):
    # nib loaded image , slice
    pixdim_xyz = img.header["pixdim"][1:4]
    print(pixdim_xyz)
    pixel_area = pixdim_xyz[0] * pixdim_xyz[1]
    print(pixel_area, "mm")
    img = img.get_data()[:, :, slice]
    pixel_value_lv = 3
    pixel_value_lv_m = 2

    np_pixdata = np.array(img)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 24))

    np_counter = np.count_nonzero(np_pixdata[(np_pixdata == pixel_value_lv)]) * pixel_area
    an_array = np.where(np_pixdata != pixel_value_lv, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    print(center_of_mass)
    img3 = Image.fromarray(an_array, 'L')
    ax1.plot(center_of_mass[1], center_of_mass[0], "or")
    ax1.imshow(img3)

    str1 = "Left Ventricle : " + str(np_counter) + "mm"
    ax1.set_title(str(str1))

    np_counter = np.count_nonzero(np_pixdata[(np_pixdata == pixel_value_lv_m)]) * pixel_area
    an_array = np.where(np_pixdata != pixel_value_lv_m, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    img4 = Image.fromarray(an_array, 'L')
    ax2.imshow(img4)
    str2 = "Myocardium : " + str(np_counter) + "mm"
    ax2.set_title(str(str2))


class Measures:
    # pixdim_xyz = image.header["pixdim"][1:4]
    # pixel_area = pixdim_xyz[0] * pixdim_xyz[1]
    pixel_area = 1
    # pix_z = img_nifti.header["pixdim"][3]
    pix_z = 10

    def calculate_values2(self, image, slice_number, pixel_value):
        image = nib.load(image)
        img = image.get_data()[:, :, slice_number]
        np_pixdata = np.array(img)
        np_counter = np.count_nonzero(np_pixdata[(np_pixdata == pixel_value)]) * self.pixel_area
        return np_counter

    def calculate_volume(self, image_list, frame):
        volume = 0
        for i in range(len(image_list)):
            # print(i, frame)
            area = self.calculate_values2(image_list[i], frame, 3)
            # print(area, "mm")
            volume += area * self.pix_z
        volume2 = 0
        # print("Frame number" + str(frame))
        for i in reversed(range(1, len(image_list))):
            # print(i, frame)
            area = self.calculate_values2(image_list[i], frame, 3)
            # print(area, "mm")
            volume2 += area * self.pix_z
        # print("------------")
        # print("Volumes:")
        # print(volume, "mm^3")
        # print(volume2, "mm^3")
        # print("average volume:")
        avg_volume = (volume2 + volume) / 2
        # print(avg_volume, "mm^3")
        # print(avg_volume * 0.001, "ml")
        return avg_volume * 0.001

    def full_volume_and_ef_calculate(self, image_list):
        image = nib.load(image_list[0])

        nr_of_frames = image.header['dim'][3]
        print('-----------')
        volume_list = []
        for frame in range(nr_of_frames):
            volume_list.append(self.calculate_volume(image_list, frame))
        return self.ejection_fraction(volume_list)

    def volume(self, image_list, nr_of_frames):
        volume_list = []
        for frame in range(0, nr_of_frames):
            volume_list.append(self.calculate_volume(image_list, frame))
        return volume_list

    def ejection_fraction(self, volume_list):
        ED = max(volume_list)
        ES = min(volume_list)
        print("ED", ED)
        print("ES", ES)
        EF = (ED - ES) / ED
        percentage_ef = EF * 100
        print(percentage_ef, "% Ejection Fraction")
        return percentage_ef, ED, ES
