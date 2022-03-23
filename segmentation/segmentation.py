from itertools import cycle

import nib as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour, morphological_geodesic_active_contour, inverse_gaussian_gradient
from scipy import ndimage
from sympy.geometry import Line
from sympy import Point as P
from shapely.geometry import LineString, LinearRing, Point
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
import math
from utilities.gifs import Gifs


class Segmentation:
    lines = LinearRing()
    snakes = []

    gifs = Gifs()

    # start_path="/content/drive/MyDrive/Dados/LV/d1/nnh/patient107_frame"
    start_path = "/content/drive/MyDrive/Dados/LV/d2/zzz/patient109_frame"
    # start_path_output = "/content/drive/MyDrive/Dados/Patient107_Output/patient107_frame"
    start_path_output = "/content/drive/MyDrive/Dados/Patient109_output/patient109_frame"

    def snake_segmentation(self, patient_str, number, frame, point_space):
        cmap = plt.cm.cool
        img_nifti_4d = nib.load(self.start_path + str(number) + "_0000.nii.gz")
        full_img = img_nifti_4d.get_data()[:, :, frame]
        pixdim_xyz = img_nifti_4d.header["pixdim"][1:4]
        pixel_area = pixdim_xyz[0] * pixdim_xyz[1]

        # alterar para mais pacientes
        nifti_image_pred = self.start_path_output + str(number) + ".nii.gz"
        img_nifti = nib.load(nifti_image_pred)
        img = img_nifti.get_data()[:, :, frame]
        np_pixdata = np.array(img)
        np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
        an_array = np.where(np_pixdata != 2, 0, np_pixdata)

        if frame < 10:
            frame = "0" + str(frame)

        img = rgb2gray(an_array)

        s = np.linspace(0, 2 * np.pi, 400)
        r = 250 + 50 * np.sin(s)
        c = 280 + 50 * np.cos(s)
        init = np.array([r, c]).T
        snake = active_contour(gaussian(img, 3),
                               init, boundary_condition="periodic",
                               alpha=0.015, beta=10, gamma=0.001, w_line=4, w_edge=8, max_iterations=5000)

        np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
        an_array = np.where(np_pixdata != 3, 0, np_pixdata)
        center_of_mass = ndimage.measurements.center_of_mass(an_array)

        img = rgb2gray(an_array)
        snake2 = active_contour(gaussian(img, 3),
                                init, boundary_condition="periodic",
                                alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
        self.snakes.append((snake, snake2, center_of_mass))

        an_array = np.where(an_array == 3, 1, 0)

        xn = snake[:, 0]
        yn = snake[:, 1]
        xn2 = snake2[:, 0]
        yn2 = snake2[:, 1]

        fig = plt.figure(figsize=(9, 9))

        ax = fig.add_subplot(111)
        plt.gray()

        ax.imshow(full_img)
        ax.plot(snake[:, 0], snake[:, 1], '-r', lw=2)
        ax.plot(snake2[:, 0], snake2[:, 1], '-r', lw=2)
        ax.axis([0, img.shape[1], img.shape[0], 0])
        ax.plot(center_of_mass[1], center_of_mass[0], "or")
        x, y = center_of_mass[1], center_of_mass[0]

        length = 50
        contour = LinearRing([(xn[j], yn[j]) for j in range(len(xn))])

        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22",
                       "#ffffff", "#000000",
                       "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22",
                       "#ffffff", "#000000",
                       "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22",
                       "#ffffff", "#000000",
                       "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22",
                       "#ffffff", "#000000"]
        res = []
        count = 0
        for i in range(len(xn2)):
            if i % point_space == 0:
                if count >= len(color_cycle):
                    count = 0
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
                y = ((m * x) + b)
                ax.plot(line, '-', color=color)
                ax.plot(intersect.x, intersect.y, 'o', color=color)
                ax.plot(xn2[i], yn2[i], 'o', color=color)
                count += 1

        color_cycle = color_cycle[0:count - 1]
        plt.show()

        return fig, contour, line, snake2, xn2, yn2, res

    def snake_segmentation_without_dots(self, patient_str, number, frame, point_space):
        cmap = plt.cm.cool
        img_nifti_4d = nib.load(self.start_path + str(number) + "_0000.nii.gz")
        full_img = img_nifti_4d.get_data()[:, :, frame]
        pixdim_xyz = img_nifti_4d.header["pixdim"][1:4]
        pixel_area = pixdim_xyz[0] * pixdim_xyz[1]

        # alterar para mais pacientes
        nifti_image_pred = self.start_path_output + str(number) + ".nii.gz"
        img_nifti = nib.load(nifti_image_pred)
        img = img_nifti.get_data()[:, :, frame]
        np_pixdata = np.array(img)
        np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
        an_array = np.where(np_pixdata != 2, 0, np_pixdata)

        if frame < 10:
            frame = "0" + str(frame)

        img = rgb2gray(an_array)

        s = np.linspace(0, 2 * np.pi, 400)
        r = 250 + 50 * np.sin(s)
        c = 280 + 50 * np.cos(s)
        init = np.array([r, c]).T
        snake = active_contour(gaussian(img, 3),
                               init, boundary_condition="periodic",
                               alpha=0.015, beta=10, gamma=0.001, w_line=4, w_edge=8, max_iterations=5000)

        np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
        an_array = np.where(np_pixdata != 3, 0, np_pixdata)
        center_of_mass = ndimage.measurements.center_of_mass(an_array)

        img = rgb2gray(an_array)
        snake2 = active_contour(gaussian(img, 3),
                                init, boundary_condition="periodic",
                                alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
        self.snakes.append((snake, snake2, center_of_mass))

    def compute_distances(self, test, patient_str, slice, point_space):
        # Test = true -> 2 frames
        # Test = false -> all frames

        color_cycle = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22", "#ffffff", "#000000",
                       "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22", "#ffffff", "#000000",
                       "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#bcbd22",
                       "#7f7f7f",
                       "#17becf", "#1f76a4", "#ff7a8e", "#2cb12c", "#f52728", "#3457bd", "#8d664b", "#e377d2",
                       "#ccfa22", "#ffffff", "#000000", ]
        color_cycle = cycle(color_cycle)
        distances = []
        img_nifti_4d = nib.load(self.start_path + str(slice) + "_0000.nii.gz")
        frame_number = img_nifti_4d.header["dim"][3]
        if test:
            frame_number = 3

        fig_list = []
        fig2_list = []
        points = []
        for i in range(1, frame_number):
            # for i in range(1,3):
            fig, c, l, s, xn, yn, res = self.snake_segmentation(patient_str, slice, i, point_space)
            fig_list.append(fig)
            fig2 = plt.figure(figsize=(7, 7))
            ax = fig2.add_subplot(111)
            plt.gray()
            x = []
            y = []
            temp_points = np.array(res)[:, 1]
            temp_points = [float(i) for i in temp_points]
            points.append(temp_points)
            for j in range(len(res)):
                ax.plot(j, res[j][1], 'o', color=res[j][0])
                if i == 1:
                    distances.append([res[j][1]])
                else:
                    distances[j].append(res[j][1])
                y.append(res[j][1])
                x.append(j)

            ax.plot(x, y, '-')
            plt.show()
            fig2_list.append((fig2, ax))
        print(len(fig2_list))
        # distances
        # Retirar mins e max de distances
        print(distances)
        for i in range(len(distances)):
            distances[i] = (round(min(distances[i]), 2), round(max(distances[i]), 2))
            # distances[i]=(round(distances[i][0],2),round(distances[i][1],2))
        print(distances)
        # BOTAR AQUI CÓDIGO PARA AGARRAR NA LISTA DE FIGURAS ADICIONAR LEGENDA E CRIAR GIF
        frame = 1
        for i in fig_list:
            ax = i.add_subplot(1, 1, 1)
            custom_lines = [Line2D([0], [0], label="min=" + str(distances[i][0]) + "   max=" + str(distances[i][1]),
                                   color=next(color_cycle), lw=4) for i in range(len(distances))]
            ax.legend(handles=custom_lines)
            i.savefig(str(frame).zfill(2) + '.png')
            frame += 1
        frame = 1
        for i in fig2_list:
            count = 0
            x = []
            ymin = []
            ymax = []
            for j in distances:
                x.append(count)
                ymin.append(j[0])
                ymax.append(j[1])
                count += 1
            ax = i[0].add_subplot(1, 1, 1)
            ax.plot(x, ymin, '-r')
            ax.plot(x, ymax, '-b')
            print(frame)
            i[0].savefig(str(frame).zfill(2) + 'graph.png')
            frame += 1

        x_axis = []
        graph = plt.figure(figsize=(10, 10))
        ax = graph.add_subplot(111)
        plt.gray()
        for i in range(len(points)):
            x_axis.append(i)
        for j in range(len(points[0])):
            y = np.array(points)[:, j]
            for i in range(len(y)):
                print("before:", y[i])
                y[i] -= distances[j][0]
                y[i] = y[i] / (distances[j][1] - distances[j][0])
                print(i)
                print("distances:", distances[j])
                print("after:", y[i])
            ax.plot(x_axis, y, '-', color=res[j][0])
        plt.show()
        print(points)

        self.gifs.save_gif_2d(patient_str + str(slice))
        self.gifs.save_gif_2d_graph()

    def get_3d(self, frame):
        fig = plt.figure()
        snakes = self.snakes
        ax = fig.gca(projection='3d')
        snake = snakes[0][0]
        snake2 = snakes[0][1]
        first_center_of_mass = (250, 250)
        print(first_center_of_mass)
        top = len(snakes) * 10
        for i in snakes:

            center_of_mass = i[2]
            print(center_of_mass)
            x, y = (abs(center_of_mass[1] - first_center_of_mass[1]), abs(center_of_mass[0] - first_center_of_mass[0]))
            print("x" + str(x) + "y" + str(y))
            # if not math.isnan(x) and not math.isnan(y):
            # snake = np.roll(i[0],[int(y),int(x)],axis=(0,1))
            # snake2 = np.roll(i[1],[int(y),int(x)],axis=(0,1))
            # else:
            snake = i[0]
            snake2 = i[1]
            if not math.isnan(x) and not math.isnan(y):
                ax.plot(np.roll(snake[:, 0], int(x)), np.roll(snake[:, 1], int(y)), '-r', lw=3, zs=top)
                ax.plot(np.roll(snake2[:, 0], int(x)), np.roll(snake2[:, 1], int(y)), '-b', lw=3, zs=top)
            else:
                ax.plot(snake[:, 0], snake[:, 1], '-r', lw=3, zs=top)
                ax.plot(snake2[:, 0], snake2[:, 1], '-b', lw=3, zs=top)
            ax.set_xlim([200, 350])
            ax.set_ylim([200, 350])

            ax.scatter(center_of_mass[1], center_of_mass[0], top)
            top -= 10
        if frame < 10:
            frame = '0' + str(frame)
        fig.savefig(str(frame) + '3d' + '.png')