import nibabel as nib
import matplotlib.pyplot as plt
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

def snake_segmentation_without_dots(image,frame):
    img = image.get_data()[:, :, frame]
    img_shape = img.shape
    print(img_shape)
    #ax.imshow(img)


    np_pixdata = np.array(img)

    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)

    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    img = rgb2gray(an_array)

    s = np.linspace(0, 2 * np.pi, 500)
    r = center_of_mass[0] + 200 * np.sin(s)
    c = center_of_mass[1] + 200 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)

    #np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    img = rgb2gray(an_array)
    snake2 = active_contour(gaussian(img, 3),
                            init, boundary_condition="periodic",
                            alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
    snake_array = np.zeros([img_shape[0]*100, img_shape[1]*100])
    print(snake)
    points_r, points_c = list(map(lambda x: round(x*100), snake[:, 0])), list(map(lambda x: round(x*100), snake[:, 1]))
    print(points_r)
    print(points_c)
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    snake_array[perimeter_r, perimeter_c] = 1
    snake_array[points_r, points_c] = 1
    snake_array[interior_r, interior_c] = 1
    points_r, points_c = list(map(lambda x: round(x*100), snake2[:, 0])), list(
        map(lambda x: round(x*100), snake2[:, 1]))
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    snake_array[perimeter_r, perimeter_c] = 2
    snake_array[points_r, points_c] = 2
    snake_array[interior_r, interior_c] = 2
    return snake_array[::100, ::100]

def segment_patient(path_to_file, path_to_output_folder):
    fig_list = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    image = nib.load(path_to_file)
    header = image.header
    print(header['dim'][3])
    for i in range(header['dim'][3]):
        try:
            print("Frame:" + str(i))
            snake = snake_segmentation_without_dots(image, i)
            fig_list.append(snake)
        except Exception as e:
            print(e)
    #print(fig_list[0].shape)
    save_array_list_to_nifti(fig_list,fig_list[0].shape[0],fig_list[0].shape[1],path_to_output_folder+'/'+patient_str, header)