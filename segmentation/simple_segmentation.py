import os

import nibabel as nib
import numpy as np
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage
from skimage.draw import polygon, polygon_perimeter

from utilities.dicom2nifticonverter import save_array_list_to_nifti

def open_nifti_image(path_to_file):
    nifti_image_pred = path_to_file
    return nib.load(nifti_image_pred)

def snake_segmentation_without_dots(image, frame, alpha=0.1, beta=1, gamma=0.001, w_line=-0.1, w_edge=4.0,
                                    max_iterations=1000, nr_of_points=500):
    img = image.get_data()[:, :, frame]
    img_shape = img.shape
    print(img_shape)
    # ax.imshow(img)

    np_pixdata = np.array(img)

    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)

    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    img = rgb2gray(an_array)

    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 200 * np.sin(s)
    c = center_of_mass[1] + 200 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=alpha, beta=beta, gamma=gamma, w_line=-1.0, w_edge=10.0,
                           max_iterations=max_iterations)

    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 200 * np.sin(s)
    c = center_of_mass[1] + 200 * np.cos(s)
    init = np.array([r, c]).T
    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    img = rgb2gray(an_array)
    snake2 = active_contour(gaussian(img, 3),
                            init, boundary_condition="periodic",
                            alpha=alpha, beta=beta, gamma=gamma, w_line=0.0, w_edge=7.5,
                            max_iterations=max_iterations)
    snake_array = np.zeros([img_shape[0] * 10, img_shape[1] * 10])
    print(snake)
    points_r, points_c = list(map(lambda x: round(x * 10), snake[:, 0])), list(
        map(lambda x: round(x * 10), snake[:, 1]))
    print(points_r)
    print(points_c)
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    snake_array[perimeter_r, perimeter_c] = 2
    snake_array[points_r, points_c] = 2
    snake_array[interior_r, interior_c] = 2
    points_r, points_c = list(map(lambda x: round(x * 10), snake2[:, 0])), list(
        map(lambda x: round(x * 10), snake2[:, 1]))
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    snake_array[perimeter_r, perimeter_c] = 3
    snake_array[points_r, points_c] = 3
    snake_array[interior_r, interior_c] = 3
    return snake_array[::10, ::10]


def segment_patient(path_to_file, path_to_output_folder):
    fig_list = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    image = nib.load(path_to_file)
    header = image.header
    affine = image.affine
    print(header['dim'][3])
    #for nr_of_points in [500,1000]:
    #for nr_of_points in [1000]:
        #for w_edge in [5,7.5,10,12.5,15,17.5,20]:
        #for w_edge in [1,2,3,4,5,6,7,8,9,10]:
    #    for w_edge in [7,7.5,8]:
            #for w_line in [-5,-4,-3,-2,-1,0,1,2,3,4,5]:
    #        for w_line in [-0.5,0,0.5]:
         #   for w_line in [-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1]:
                #for max_iterations in [500,1000]:
    #            for max_iterations in [1000]:
          #      for max_iterations in [250, 500, 1000]:
    try:
                        for i in range(header['dim'][3]):

                                print("Frame:" + str(i))
                                try:
                                    snake = snake_segmentation_without_dots(image, i)
                                    fig_list.append(snake)
                                except Exception as e:
                                    print(e)
                        file_path = path_to_output_folder + '/' + f'10.0,-1.0,1000,500-7.5,0.0,1000,500'
                        if not os.path.exists(file_path):
                            os.makedirs(file_path)
                        save_array_list_to_nifti(fig_list, fig_list[0].shape[0],
                                                 fig_list[0].shape[1],
                                                 file_path + '/' + patient_str,
                                                 affine, header.copy())
    except Exception as e:
        print(e)

for filename in os.listdir('/Users/antonioneto/Downloads/testing_output/OUTPUT_TESTING'):
    if filename.endswith('.nii'):
        segment_patient(f'/Users/antonioneto/Downloads/testing_output/OUTPUT_TESTING/{filename}',
                        '/Users/antonioneto/Downloads/testing_output')




"""
alpha=0.1, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=1000)
alpha = 0.1

beta and gamma set alpha set
            FIRST TEST              SECOND TEST
w_edge    [0,5,10,15,20]  ----->     [5,10,15] ----> [5,6,7,8,9,10] ---> LV [7.8,7.9,8.0,8.1,8.2] MYO [1,2,3,4,5] --> LV 8.1 MYO [3.0,3.1,3.2,3.3] ---> MYO 3.1
w_line    [-5.0,0.0,5.0]  ----->  [-2.0,-1.0,0.0,1.0] ---> [-0.5,-0.25,0.0,0.25,0.5] --> LV[-0.6 ,-0.5, -0.4] MYO[-0.1,-0.15,-0.2,-0.25,-0.3] -> MYO [-0.225,-0.25,-0.275] -> LV -0.5  ---> MYO -0.12 -> -0.09
max_iterations 1000   --------->  [1000,10000] -> [500,1000] -> 500
"""
