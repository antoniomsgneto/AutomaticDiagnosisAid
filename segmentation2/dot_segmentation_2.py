import itertools
import os
from dataclasses import dataclass
from enum import Enum
from time import sleep

import imageio
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage

from sympy.geometry import Line
from sympy import Point as P
from matplotlib.lines import Line2D
from shapely.geometry import LineString, LinearRing, Point
from scipy.spatial import distance as spatial_distance

from PIL import Image
import io

from utilities.video_presenter import show_gif_loop
class SegmentationLevel(Enum):
    MYO = 2
    LV = 3
    RV = 1
snake_colors = {3: '-r', 2: '-b'}

color_cycle = itertools.cycle(
    ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
     "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2", "#ccfa22",
     "#ffffff", "#000000"])


@dataclass
class SnakeSettings:
    nr_of_points: int
    max_iterations: int
    alpha: float
    beta: float
    gamma: float
    w_line: float
    w_edge: float


settings = {
    SegmentationLevel.LV: SnakeSettings(
        nr_of_points=100,
        max_iterations=500,
        alpha=0,
        beta=1,
        gamma=0.001,
        w_line=-1.0,
        w_edge=10
    ),
    SegmentationLevel.MYO: SnakeSettings(
        nr_of_points=100,
        max_iterations=500,
        alpha=0,
        beta=1,
        gamma=0.001,
        w_line=-1.0,
        w_edge=10
    )
}

snakes = {}
fig_list = []

def snake_segmentation_dots(
    path_to_file
):
    image = nib.load(path_to_file)
    pixdim_xyz = image.header["pixdim"][1:4]
    pixel_area = pixdim_xyz[0] * pixdim_xyz[1]
    for slice in range(0, len(image.get_data()[:, :])):
        try:
            img = image.get_data()[:, :, slice]
            np_pixdata = np.array(img)
            np_pixdata = np.where(np_pixdata == SegmentationLevel.RV.value, 0, np_pixdata)
            an_array = np.where(np_pixdata < SegmentationLevel.LV.value, 0, np_pixdata)
            center_of_mass = ndimage.measurements.center_of_mass(an_array)
            print("Applying Snakes")
            for seg_level in SegmentationLevel:
                if seg_level.value != SegmentationLevel.RV.value:
                    snakes[seg_level.value] = snake_seg(np_pixdata, center_of_mass, seg_level)

            fig = plt.figure(figsize=(18, 9))
            ax = fig.add_subplot(121)
            ax2 = fig.add_subplot(222)
            ax.imshow(img)
            # Print Raw Snakes
            for index in snakes:
                ax.plot(snakes[index][:, 1], snakes[index][:, 0], snake_colors[index], lw=3)

            print("Applying Dots")
            dots = add_dots(pixel_area, center_of_mass, snakes)

            custom_lines = []

            print("Building Graphs")

            # Print Dots
            count = 0
            for d, do, distance in dots:
                color = next(color_cycle)
                ax.plot(d[0], d[1], 'o', color=color)
                ax.plot(do[0], do[1], 'o', color=color)
                custom_lines.append(Line2D([0], [0], label=str(round(distance, 2)) + "mm",
                                           color=color, lw=4))
                ax2.plot(count, distance, 'o', color=color)
                count += 1
            fig.legend(title='Thickness', handles=custom_lines, loc='upper right')

            save_fig_to_list(fig)
        except Exception as e:
            print(e)


def snake_seg(np_pixdata, center_of_mass, seg_level: SegmentationLevel):
    snake_settings = settings[seg_level]
    an_array = np.where(np_pixdata < seg_level.value, 0, np_pixdata)
    img = rgb2gray(an_array)
    s = np.linspace(0, 2 * np.pi, snake_settings.nr_of_points)
    r = center_of_mass[0] + 350 * np.sin(s)
    c = center_of_mass[1] + 350 * np.cos(s)
    init = np.array([r, c]).T
    return active_contour(gaussian(img, 3),
                          init, boundary_condition="periodic",
                          alpha=snake_settings.alpha, beta=snake_settings.beta, gamma=snake_settings.gamma,
                          w_line=snake_settings.w_line, w_edge=snake_settings.w_edge,
                          max_iterations=snake_settings.max_iterations)


def add_dots(pixel_area, center_of_mass, snake_list):
    res = []

    inner_snake = snake_list[SegmentationLevel.LV.value]
    outter_snake = snake_list[SegmentationLevel.MYO.value]

    xn2 = outter_snake[:, 1]  # inner
    yn2 = outter_snake[:, 0]
    xn = inner_snake[:, 1]  # outter
    yn = inner_snake[:, 0]

    contour = LinearRing([(xn[j], yn[j]) for j in range(len(xn))])
    for i in range(len(xn2)):
        if i % 5 == 0:
            l = Line(P(xn2[i - 1], yn2[i - 1], evaluate=False), P(xn2[i + 1], yn2[i + 1], evaluate=False))
            lin = l.perpendicular_line(P(xn2[i], yn2[i]))
            l = lin.coefficients
            m = l[0] / (-l[1])
            b = l[2] / (-l[1])
            line = LineString([(0, b), (400, (m * 400) + b)])
            intersect = line.intersection(contour)
            distance0 = intersect[0].distance(Point(xn2[i], yn2[i])) * pixel_area
            distance1 = intersect[1].distance(Point(xn2[i], yn2[i])) * pixel_area
            if distance0 < distance1:
                distance = distance0
                intersect = intersect[0]
            else:
                distance = distance1
                intersect = intersect[1]
            try:
                center_distance = spatial_distance.euclidean(center_of_mass, (yn[i], xn[i]))
                res.append([(intersect.x, intersect.y), (xn2[i], yn2[i]), distance])
            except Exception as e:
                print(e)
    return res


def save_fig_to_list(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    # Open the image from the buffer and append it to the list
    image = Image.open(buf)
    fig_list.append(image)


#snake_segmentation_dots('/Users/antonioneto/Downloads/OUTPUT_TESTING/patient150_frame12.nii')
snake_segmentation_dots('/Users/antonioneto/Antonio/tese/Dados/OUTPUT_DIRECTORY_3D/Patient_CMD7_10.nii')
imageio.mimsave('/Users/antonioneto/Downloads/animation.gif', fig_list)
# show_gif_loop('/Users/antonioneto/Downloads/animation.gif')

import subprocess

try:
    cmd = "open -a 'google chrome' /Users/antonioneto/Downloads/animation.gif"
    subprocess.run(cmd, shell=True)
finally:
    cmd = "rm /Users/antonioneto/Downloads/animation.gif"
    subprocess.run(cmd, shell=True)
