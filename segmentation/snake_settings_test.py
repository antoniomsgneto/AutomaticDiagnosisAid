# Import Dependencies
import itertools
import os
from dataclasses import dataclass
from enum import Enum
import imageio
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage import draw
from scipy import ndimage

from medpy.metric.binary import hd, dc
from sympy.geometry import Line
from sympy import Point as P
from matplotlib.lines import Line2D
from shapely.geometry import LineString, LinearRing, Point
from scipy.spatial import distance as spatial_distance

from PIL import Image
import io

from acdc.metrics import metrics
from segmentation.dot_segmentation_2 import SegmentationLevel

# Setting Variables

NR_OF_POINTS = 1000
NR_OF_INTEREST_POINTS = 20

point_interval = NR_OF_POINTS / NR_OF_INTEREST_POINTS

# Load image nifti and choose slice to display
SLICE = 1
nifti_image_pred = '/Users/antonioneto/Downloads/training/patient001/patient001_frame12_gt.nii.gz'  # Reach out in case you need a file to test
img_nifti = nib.load(nifti_image_pred)
background = img_nifti.get_data()[:, :, SLICE]
pixdim_xyz = img_nifti.header["pixdim"][1:4]
pixel_area = pixdim_xyz[0] * pixdim_xyz[1]
voxel_size = img_nifti.header.get_zooms()
COUNT = 0

def snake_segmentation(image_array, gamma, nr_of_points, gaussian_sigma, pixel_area, w_edge=2):
    # Define a segmentation map for MYO and LV
    myo_seg = np.where(image_array < SegmentationLevel.MYO.value, 0, image_array)
    lv_seg = np.where(image_array < SegmentationLevel.LV.value, 0, image_array)

    # Calculate the area of 'myo_seg'
    myo_area = np.sum(myo_seg != 0)

    # Calculate the area of 'lv_seg'
    lv_area = np.sum(lv_seg != 0)

    #if lv_area < 100:
        #raise Exception("A")

    # if lv_area < 50:
    # nr_of_points = 50
    # gamma = 0.01

    # if myo_area < 200:
    #  nr_of_points = 70

    # Print the results
    print("Area of myo_seg:", myo_area)
    print("Area of lv_seg:", lv_area)
    # Define initial shape of snake
    # Define center of mass for MYO
    center_of_mass = ndimage.measurements.center_of_mass(myo_seg)
    # nr_of_points = 500
    # gaussian_sigma = 2
    # gamma = 0.0002
    nr_of_points = 500
    gaussian_sigma = 2
    gamma = 0.0002

    # Define initial shape of snake
    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 100 * np.sin(s)
    c = center_of_mass[1] + 100 * np.cos(s)
    init = np.array([r, c]).T

    # Execute snake segmentation for MYO and LV
    myo_snake = active_contour(gaussian(myo_seg, gaussian_sigma, preserve_range=False),
                               init,
                               #
                               # alpha=0.1,
                               # beta=5,
                               w_line=-0.1,
                               w_edge=2,
                               gamma=gamma,
                               # max_num_iter=1000
                               )

    nr_of_points = 100
    gaussian_sigma = 1
    gamma = 0.005
    # w_edge = 3
    #if True:  # lv_area < 50:
        #nr_of_points = 75
    #    gamma = 0.005
        # gaussian_sigma=1

    s = np.linspace(0, 2 * np.pi, nr_of_points)
    r = center_of_mass[0] + 100 * np.sin(s)
    c = center_of_mass[1] + 100 * np.cos(s)
    init = np.array([r, c]).T

    lv_snake = active_contour(
        gaussian(lv_seg, gaussian_sigma, preserve_range=False),
        init,
        # alpha=0.1,
        # beta=2,
        w_line=-0.1,
        w_edge=2,
        gamma=gamma,
        # max_num_iter=1000
    )
    myo_mask = np.zeros_like(image_array, dtype=np.bool)
    lv_mask = np.zeros_like(image_array, dtype=np.bool)
    rr, cc = draw.polygon(myo_snake[:, 0], myo_snake[:, 1])
    myo_mask[rr, cc] = 1
    rr, cc = draw.polygon(lv_snake[:, 0], lv_snake[:, 1])
    lv_mask[rr, cc] = 1
    return [metrics((lv_seg != 0).astype(np.bool), lv_mask, voxel_size),
            metrics((myo_seg != 0).astype(np.bool), myo_mask, voxel_size)]
    # return metrics((myo_seg != 0).astype(np.bool),myo_mask,voxel_size)


# stats = [[0.005,100,1], [0.0002,500,1], [0.00005,1000,1],[0.005,100,2],[0.0002,500,2],[0.00005,1000,2],[0.005,100,3],[0.0002,500,3],[0.00005,1000,3]]
# stats = [[0.005,100,1], [0.0002,500,1], [0.00005,1000,1],[0.005,100,2],[0.0002,500,2],[0.00005,1000,2],[0.005,100,3],[0.0002,500,3],[0.00005,1000,3]]
stats = [[0.005, 100, 1]]
directory_path = '/Users/antonioneto/Downloads/training/extracted'
for SLICE in [1]:  # [1,3,5]:
    # print(f"SLICE: {SLICE}")
    for stat in stats:
        # for w_edge in [2]:
        results_lv = []
        results_myo = []
        for filename in os.listdir(directory_path):
            nifti_image_pred = f'{directory_path}/{filename}'  # Reach out in case you need a file to test
            img_nifti = nib.load(nifti_image_pred)
            try:
                print(filename)
                shape = img_nifti.shape
                #SLICE = int((shape[2] - 1  )/2)# Apical SLICe/
                #SLICE = (shape[2] - 1)  # Apical SLICe/
                background = img_nifti.get_data()[:, :, SLICE]
                print(SLICE)
                pixdim_xyz = img_nifti.header["pixdim"][1:4]
                pixel_area = pixdim_xyz[0] * pixdim_xyz[1]
                # print(pixel_area)
                voxel_size = img_nifti.header.get_zooms()
                if np.any(background == SegmentationLevel.MYO.value) and np.any(
                        background == SegmentationLevel.LV.value):
                    res = snake_segmentation(background, stat[0], stat[1], stat[2], pixel_area)
                    # print(res[3:6])

                    results_lv.append(res[0])
                    results_myo.append(res[1])
                    # results.append(w_edge_test(background,w_edge))
                else:
                    background = img_nifti.get_data()[:, :, SLICE - 1]
                    if np.any(background == SegmentationLevel.MYO.value) and np.any(
                            background == SegmentationLevel.LV.value):
                        res = snake_segmentation(background, stat[0], stat[1], stat[2], pixel_area)
                        # print(res[3:6])
                        results_lv.append(res[0])
                        results_myo.append(res[1])
            except Exception as e:
                COUNT += 1
                print(f"{filename} doesnt have that slice")
                print(e)
        print(len(results_lv))
        averages_lv = [sum(col) / len(col) for col in zip(*results_lv)]
        averages_myo = [sum(col) / len(col) for col in zip(*results_myo)]
        results_lv = []
        results_myo = []

        # Print the averages
        for i, avg in enumerate(averages_lv):

            # print(f"w_edge:{w_edge}")
            if avg != 0.0:
                print("LV")
                print(stat)
                print(f"Average for index {i}: {avg}")
        for i, avg in enumerate(averages_myo):

            # print(f"w_edge:{w_edge}")
            if avg != 0.0:
                print("MYO")
                print(stat)
                print(f"Average for index {i}: {avg}")
