import csv
import pickle
from enum import Enum
from time import sleep

import nibabel as nib
import numpy as np
from medpy.metric import dc
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage
from skimage.draw import polygon, polygon_perimeter
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB


class SegmentationLevel(Enum):
    MYO = 3
    LV = 2


seg_level_options = [SegmentationLevel.MYO, SegmentationLevel.LV]
alpha_options = [0, 1, 2]
beta_options = [0, 1, 2]
gamma_options = [0.001, 0.01, 0.1]
w_line_options = [-2.0, -1.0, 0.0, 1.0]
w_edge_options = [0, 5, 10, 15, 20]
max_iterations_options = [100, 250, 500]
nr_of_points_options = [100, 250, 500]


def call_snake(image, slice, alpha, beta, gamma, w_line, w_edge, max_iterations, nr_of_points):
    print(image, slice, alpha, beta, gamma, w_line, w_edge, max_iterations, nr_of_points)
    image = nib.load(image)
    img = image.get_data()[:, :, slice]
    img_shape = img.shape
    np_pixdata = np.array(img)

    snake_M = snake_segmentation_without_dots(np_pixdata, SegmentationLevel.MYO, alpha, beta, gamma, w_line, w_edge,
                                              max_iterations, nr_of_points)
    snake_L = snake_segmentation_without_dots(np_pixdata, SegmentationLevel.LV, alpha, beta, gamma, w_line, w_edge,
                                              max_iterations, nr_of_points)

    filled_snake_L = snake_fill(img_shape, snake_L, SegmentationLevel.LV, None)

    filled_snake_M = snake_fill(img_shape, snake_M, SegmentationLevel.MYO, filled_snake_L)

    return filled_snake_M[::10, ::10]


def snake_segmentation_without_dots(
    image, segmentation_level, alpha, beta, gamma, w_line, w_edge, max_iterations, nr_of_points
):
    np_pixdata = image
    an_array = np.where(np_pixdata < segmentation_level.value, 0, np_pixdata)

    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    img = an_array
    # plt.imshow(an_array)
    # plt.show()
    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 200 * np.sin(s)
    c = center_of_mass[1] + 200 * np.cos(s)

    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=alpha, beta=beta, gamma=gamma, w_line=w_line, w_edge=w_edge,
                           max_iterations=max_iterations)
    return snake


def snake_fill(img_shape, snake, segmentation_level, init_array):
    if init_array is None:
        snake_array = np.zeros([img_shape[0] * 10, img_shape[1] * 10])
    else:
        snake_array = init_array

    points_r, points_c = list(map(lambda x: round(x * 10), snake[:, 0])), list(
        map(lambda x: round(x * 10), snake[:, 1]))
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    try:
        snake_array[perimeter_r, perimeter_c] = segmentation_level.value
        snake_array[points_r, points_c] = segmentation_level.value
        snake_array[interior_r, interior_c] = segmentation_level.value
    except Exception as e:
        print(f"SNAKE EXCEEDES SIZE OF IMAGE {e}")
    return snake_array


def dice_score(arr1, arr2, segmentation_level: SegmentationLevel):
    array1 = np.where(arr1 != segmentation_level.value, 0, arr1)
    array2 = np.where(arr2 != segmentation_level.value, 0, arr2)
    #cmap1 = 'Reds'
    #cmap2 = 'Blues'
    # plt.imshow(array1, cmap=cmap1, alpha=0.8)
    # plt.imshow(array2, cmap=cmap2, alpha=0.4)
    # plt.show()
    dice = dc(array1, array2)
    return dice


def __load_file(path_to_file: str, frame):
    return np.array(nib.load(path_to_file).get_data()[:, :, frame])


def run():
    train_data = []

    path_to_train = '/Users/antonioneto/Downloads/OUTPUT_TESTING/train.pickle'
    path_to_file = '/Users/antonioneto/Downloads/OUTPUT_TESTING/patient150_frame12.nii'
    path_to_ground_truth = '/Users/antonioneto/Downloads/testing2/patient150_frame12_gt.nii'
    train_file = open('/Users/antonioneto/Downloads/train_data.csv', 'w+', newline='')
    write = csv.writer(train_file)
    write.writerow(
        [   'patient_str',
            'slice',
            'alpha',
            'beta',
            'gamma',
            'w_line',
            'w_edge',
            'max_iterations',
            'nr_of_points',
            'seg_level',
            'score_seg'
            ])
    write.writerow(
        ['patient_str',
         'slice',
         'alpha',
         'beta',
         'gamma',
         'w_line',
         'w_edge',
         'max_iterations',
         'nr_of_points',
         'seg_level',
         'score_seg'
         ])
    for slice in range(1, 9):
        for alpha in alpha_options:
            for beta in beta_options:
                for gamma in gamma_options:
                    for w_line in w_line_options:
                        for w_edge in w_edge_options:
                            for max_iterations in max_iterations_options:
                                for nr_of_points in nr_of_points_options:
                                    try:
                                        snake = call_snake(
                                            path_to_file,
                                            slice,
                                            alpha=alpha,
                                            beta=beta,
                                            gamma=gamma,
                                            w_line=w_line,
                                            w_edge=w_edge,
                                            max_iterations=max_iterations,
                                            nr_of_points=nr_of_points
                                        )

                                        np_pixdata = __load_file(path_to_file, slice)
                                        for seg_level in seg_level_options:
                                            score_seg = dice_score(np_pixdata, snake, seg_level)
                                            with open(path_to_train, 'ab') as f:
                                                write.writerow([
                                                    'patient150_frame12',
                                                    slice,
                                                    alpha,
                                                    beta,
                                                    gamma,
                                                    w_line,
                                                    w_edge,
                                                    max_iterations,
                                                    nr_of_points,
                                                    seg_level,
                                                    score_seg
                                                ])
                                    except:
                                        print(f'couldnt process this slice {slice}')
            # for seg_level in SegmentationLevel:
            # print(f'---------------{seg_level}---------------------------')
            # print(f'Segmentation Unet - Snake {dice_score(np_pixdata, snake, seg_level)}')
            # print(f'Ground Truth - Snake {dice_score(np_pixdata_gt, snake, seg_level)}')
    # with open(path_to_train,'ab') as f:
    # pickle.dump(train_data, f)

    # nb = GaussianNB()
    # X_train = np.array(train_data)[:, :-1]
    # y_train = np.array(train_data)[:, -1]
    # nb.fit(X_train, y_train)

a = "print Tests"
run()


