import os
from enum import Enum

import nibabel as nib
import numpy as np
from medpy.metric import dc
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage
from skimage.draw import polygon, polygon_perimeter

class SegmentationLevel(Enum):
    MYO = 3
    LV = 2


def snake_segmentation_without_dots(
    image, slice, segmentation_level, alpha, beta, gamma, w_line, w_edge, max_iterations, nr_of_points
):
    image = nib.load(image)
    img = image.get_data()[:, :, slice]
    img_shape = img.shape
    np_pixdata = np.array(img)

    an_array = np.where(np_pixdata < segmentation_level.value, 0, np_pixdata)

    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    img = rgb2gray(an_array)
    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 200 * np.sin(s)
    c = center_of_mass[1] + 200 * np.cos(s)
    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 200 * np.sin(s)
    c = center_of_mass[1] + 200 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=alpha, beta=beta, gamma=gamma, w_line=w_line, w_edge=w_edge,
                           max_iterations=max_iterations)
    snake_array = np.zeros([img_shape[0] * 10, img_shape[1] * 10])
    points_r, points_c = list(map(lambda x: round(x * 10), snake[:, 0])), list(
        map(lambda x: round(x * 10), snake[:, 1]))
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    try:
        snake_array[perimeter_r, perimeter_c]= segmentation_level.value
        snake_array[points_r, points_c] = segmentation_level.value
        snake_array[interior_r, interior_c] = segmentation_level.value
    except:
        print("SNAKE EXCEEDES SIZE OF IMAGE")

    return snake_array[::10, ::10]


def dice_score(arr1, arr2, segmentation_level: SegmentationLevel):
    array1 = np.where(arr1 != segmentation_level.value, 0, arr1)
    array2 = np.where(arr2 != segmentation_level.value, 0, arr2)
    dice = dc(array1, array2)
    return dice


def __load_file(path_to_file: str, frame):
    return np.array(nib.load(path_to_file).get_data()[:, :, frame])


def run():
    path_to_file = '/Users/antonioneto/Downloads/OUTPUT_TESTING/patient150_frame12.nii'
    path_to_ground_truth = '/Users/antonioneto/Downloads/testing2/patient150_frame12_gt.nii'
    snake = snake_segmentation_without_dots(
        path_to_file, 7, SegmentationLevel.LV,
        alpha=0.1,
        beta=1,
        gamma=0.001,
        w_line=-1.0,
        w_edge=10,
        max_iterations=400,
        nr_of_points=400
    )
    np_pixdata_gt = __load_file(path_to_ground_truth, 7)
    np_pixdata = __load_file(path_to_file, 7)
    for seg_level in SegmentationLevel:
        print(f'---------------{seg_level}---------------------------')
        print(f'Segmentation Unet - Snake {dice_score(np_pixdata, snake, seg_level)}')
        print(f'Ground Truth - Snake {dice_score(np_pixdata_gt, snake, seg_level)}')


run()
