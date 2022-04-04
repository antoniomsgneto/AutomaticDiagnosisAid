import imageio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage

from sympy.geometry import Line
from sympy import Point as P
from shapely.geometry import LineString, LinearRing,Point


def snake_segmentation_with_dots(path_to_file, patient_str, frame, snake_directory):
    nifti_image_pred = path_to_file
    print(nifti_image_pred)
    img_nifti = nib.load(nifti_image_pred)
    img = img_nifti.get_data()[:, :, frame]
    print(img.shape)
    pixel_area = 1
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    np_pixdata = np.array(img)
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    img = rgb2gray(an_array)

    s = np.linspace(0, 2 * np.pi, 500)
    r = 250 + 50 * np.sin(s)
    c = 280 + 50 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)

    img = rgb2gray(an_array)
    snake2 = active_contour(gaussian(img, 3),
                            init, boundary_condition="periodic",
                            alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
    ax.plot(snake[:, 1], snake[:, 0], '-r', lw=3)
    ax.plot(snake2[:, 1], snake2[:, 0], '-b', lw=3)
    ax.plot(center_of_mass[1], center_of_mass[0], "or")
    ax.axis([0, img.shape[1], img.shape[0], 0])

    yn = snake[:, 0]
    xn = snake[:, 1]
    yn2 = snake2[:, 0]
    xn2 = snake2[:, 1]
    contour = LinearRing([(xn[j], yn[j]) for j in range(len(xn))])

    color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
                   "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2", "#ccfa22",
                   "#ffffff", "#000000",
                   "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
                   "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2", "#ccfa22",
                   "#ffffff", "#000000",
                   "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
                   "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2", "#ccfa22",
                   "#ffffff", "#000000",
                   "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
                   "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2", "#ccfa22",
                   "#ffffff", "#000000"]
    res = []
    count = 0
    res_list = []
    for i in range(len(xn2)):
        if i % 20 == 0:
            color = color_cycle[count]
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
            res.append((color_cycle[count], distance, i))
            x = np.linspace(90, 140)
            res_list.append({'point': (xn2[i], yn2[i]), 'm': m, 'b': b})
            ax.plot(line, '-', color=color)
            ax.plot(intersect.x, intersect.y, 'o', color=color)

            ax.plot(xn2[i], yn2[i], 'o', color=color)
            count += 1

    fig_name = snake_directory + '/' + str(patient_str) + str(frame) + '_dots.png'
    fig.savefig(fig_name)

    fig2 = plt.figure(figsize=(7, 7))
    ax = fig2.add_subplot(111)
    plt.gray()
    count = 0

    return fig_name


def segment_patient(path_to_file, path_to_output_folder):
    fig_list = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    for i in range(30):
        try:
            fig_list.append(snake_segmentation_with_dots(path_to_file, patient_str, i, path_to_output_folder))
        except Exception as e:
            print(e)
    save_gif_2d(path_to_output_folder+'/'+patient_str, fig_list)


def save_gif_2d(patient_str, fig_list):
    with imageio.get_writer(patient_str + '.gif', mode='I') as writer:
        for fig in fig_list:
            image = imageio.imread(fig)
            writer.append_data(image)
