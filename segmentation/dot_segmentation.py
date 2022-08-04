import os

import itertools
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
from matplotlib.lines import Line2D
from shapely.geometry import LineString, LinearRing, Point
from scipy.spatial import distance as spatial_distance

#cada nifti é enviada para esta função para serem colocados 


def snake_segmentation_with_dots(path_to_original, path_to_file, patient_str, frame, snake_directory):
    nifti_image_pred = path_to_file
    print(nifti_image_pred)

    img_nifti = nib.load(nifti_image_pred)
    pixdim_xyz = img_nifti.header["pixdim"][1:4]
    pixel_area = pixdim_xyz[0] * pixdim_xyz[1]
    img = img_nifti.get_data()[:, :, frame]
    print(img.shape)

    try:
        print(path_to_original)
        background = nib.load(path_to_original).get_data()[:, :, frame]
    except:
        background = img

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(background)
    np_pixdata = np.array(img)
    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    img = rgb2gray(an_array)

    s = np.linspace(0, 2 * np.pi, 500)
    r = 250 + 100 * np.sin(s)
    c = 280 + 100 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)

    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
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

    color_cycle = itertools.cycle(["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
                   "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2", "#ccfa22",
                   "#ffffff", "#000000"])
    res = []
    count = 0
    res_list = []
    xn2 = snake[:, 1]
    yn2 = snake[:, 0]
    xn = snake2[:, 1]
    yn = snake2[:, 0]

    contour = LinearRing([(xn[j], yn[j]) for j in range(len(xn))])
    for i in range(len(xn2)):
        if i % 5 == 0:
            color = next(color_cycle)
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
            print("---------------------------------------")
            print(intersect)
            print(distance)
            center_distance = spatial_distance.euclidean(center_of_mass, (yn[i], xn[i]))
            res.append((i, color, distance, center_distance))
            print(res)
            print("CENTER DISTANCE:" + str(center_distance))
            x = np.linspace(90, 140)
            # ax.plot(l,'-',color=color)
            y = ((m * x) + b)
            res_list.append({'point': (xn2[i], yn2[i]), 'm': m, 'b': b})
            # intersect=lin.intersection(contour)
            # distance = intersect.distance(Point(xn[i],yn[i]))*pixel_area
            # ax.plot(line, '-', color=color)
            ax.plot(intersect.x, intersect.y, 'o', color=color)
            ax.plot(xn2[i], yn2[i], 'o', color=color)
            count += 1

    custom_lines = [Line2D([0], [0], label=str(round(value, 2)) + "mm",
                           color=color, lw=4) for num, color, value, _ in res]
    custom_center_lines = [Line2D([0], [0], label=str(round(value, 2)) + "mm",
                           color=color, lw=4) for num, color, _, value in res]
    legend1 = plt.legend(title='Thickness', handles=custom_lines, loc='upper right')
    legend2 = plt.legend(title='Center Distance', handles=custom_center_lines, loc='upper left')
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    fig_name = snake_directory + '/' + str(patient_str) + str(frame) + '.png'
    fig.savefig(fig_name)
    print("SAVED FIG" + fig_name)
    return fig_name, res


def segment_patient(path_to_original, path_to_file, path_to_output_folder):
    fig_list = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    distances = []
    for i in range(40):
        try:
            fig_name, res = snake_segmentation_with_dots(path_to_original, path_to_file, patient_str, i,
                                                         path_to_output_folder)
            distances = distances + res
            fig_list.append(fig_name)
        except Exception as e:
            print(e)
    save_gif_2d(path_to_output_folder + '/' + patient_str, fig_list)
    #EXPORTAR OU DAR RETURN E FAZER SCRIPT PARA ORGANIZAR EM FICHEIRO. PARA TER PELAS SLICES
    distances_by_point = export_graph(distances, path_to_output_folder, patient_str)
    distances_by_frame = distances
    for file in fig_list:
        os.remove(file)
    return distances


def save_gif_2d(patient_str, fig_list):
    with imageio.get_writer(patient_str + '.gif', mode='I') as writer:
        for fig_name in fig_list:
            image = imageio.imread(fig_name)
            writer.append_data(image)


def export_graph(res, snake_directory, patient_str):
    #ADICIONADO PELO JOÃO
    print("DISTANCES: \n")
    print(res)
    graph = plt.figure(figsize=(10, 10))
    ax = graph.add_subplot(111)
    res_dict = {}
    for item in res:
        try:
            res_dict[(item[0], item[1])].append(item[2])
        except KeyError:
            res_dict[(item[0], item[1])] = [item[2]]
    plt.gray()
    for key in res_dict.keys():
        x_axis = list(range(0, len(res_dict[key])))
        ax.plot(x_axis, res_dict[key], '-', color=key[1])
    fig_name = snake_directory + '/' + str(patient_str) + 'graph.png'
    graph.savefig(fig_name)
    return res_dict
