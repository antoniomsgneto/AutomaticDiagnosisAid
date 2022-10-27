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
import seaborn as sns


#cada nifti é enviada para esta função para serem colocados 


def snake_segmentation_with_dots(path_to_original, path_to_file, patient_str, frame, snake_directory, colors):
    
    #adicionado pelo João
    ready_for_cluster = []
    intersections = []
    
    
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

    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)

    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    img = rgb2gray(an_array)

    s = np.linspace(0, 2 * np.pi, 500)
    r = center_of_mass[0] + 30 * np.sin(s)
    c = center_of_mass[1] + 30 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=0.1, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=10000)

    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
    an_array = np.where(np_pixdata != 3, 0, np_pixdata)

    img = rgb2gray(an_array)
    snake2 = active_contour(gaussian(img, 3),
                            init, boundary_condition="periodic",
                            alpha=0.1, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
    ax.plot(snake[:, 1], snake[:, 0], '-r', lw=3)
    ax.plot(snake2[:, 1], snake2[:, 0], '-b', lw=3)
    ax.plot(center_of_mass[1], center_of_mass[0], "or")
    ax.axis([0, img.shape[1], img.shape[0], 0])
    ready_for_cluster.append(background)
    ready_for_cluster.append(snake)


    
    color_cycle = itertools.cycle(colors)
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
            intersections.append(intersect)
            ax.plot(intersect.x, intersect.y, 'o', color=color)
            ax.plot(xn2[i], yn2[i], 'o', color=color)
            count += 1
    ready_for_cluster.append(intersections)
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

    #adicionado João
    fig_name2 = snake_directory + '/' + str(patient_str) + str(frame) + 'CLUSTER.png'
    ready_for_cluster.append(fig_name2)

    return fig_name, res, ready_for_cluster


def segment_patient(path_to_original, path_to_file, path_to_output_folder, colors):
    fig_list = []
    fig_list_cluster = []
    all_cluster_info = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    distances = []
    for i in range(7,20):
        try:
            fig_name, res, cluster_info = snake_segmentation_with_dots(path_to_original, path_to_file, patient_str, i,
                                                         path_to_output_folder, colors)
            distances = distances + res
            
            #adicionado João
            fig_list_cluster.append(cluster_info[3])
            all_cluster_info.append(cluster_info)
            
            fig_list.append(fig_name)
        except Exception as e:
            print(e)
    save_gif_2d(path_to_output_folder + '/' + patient_str, fig_list)
    #EXPORTAR OU DAR RETURN E FAZER SCRIPT PARA ORGANIZAR EM FICHEIRO. PARA TER PELAS SLICES
    distances_by_point = export_graph(distances, path_to_output_folder, patient_str)
    distances_by_frame = distances
    for file in fig_list:
        os.remove(file)
    return distances, all_cluster_info, fig_list_cluster


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




def plot_graphs(cluster_info):
    print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")

    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.gray()

    ax.imshow(cluster_info[0])
    
    snake = cluster_info[1]
    
    xn2 = snake[:, 1]
    yn2 = snake[:, 0]
    colors = itertools.cycle(get_pallete(100))

    for i in range(len(xn2)):
        
        if i % 5 == 0:
                
            color = next(colors)
            
            ax.plot(cluster_info[2][int(i/5)].x, cluster_info[2][int(i/5)].y, 'o', color=color)
            ax.plot(xn2[i], yn2[i], 'o', color=color)
    fig.savefig(cluster_info[3])

def get_pallete(n):
  return sns.color_palette('magma', n).as_hex()

