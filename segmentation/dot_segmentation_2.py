import itertools
from dataclasses import dataclass
from enum import Enum
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


def calculate_thickening(max_thickness: float, min_thickness: float) -> float:
    return (max_thickness - min_thickness) / max_thickness


class SegmentationLevel(Enum):
    MYO = 2
    LV = 3
    RV = 1


snake_colors = {3: '-r', 2: '-b'}


def generate_colors(number_of_colors):
    return list(plt.cm.get_cmap('tab20')(np.linspace(0, 1, number_of_colors)))


color_cycle = itertools.cycle(generate_colors(20))
NR_OF_POINTS = 400
NR_OF_INTEREST_POINTS = 20
POINT_INTERVAL = NR_OF_POINTS / NR_OF_INTEREST_POINTS


@dataclass
class SnakeSettings:
    nr_of_points: int
    max_iterations: int
    alpha: float
    beta: float
    gamma: float


settings = {
    SegmentationLevel.LV: SnakeSettings(
        nr_of_points=NR_OF_POINTS,
        max_iterations=500,
        alpha=0.015,
        beta=10,
        gamma=0.001,
    ),
    SegmentationLevel.MYO: SnakeSettings(
        nr_of_points=NR_OF_POINTS,
        max_iterations=500,
        alpha=0.015,
        beta=10,
        gamma=0.001,
    )
}

fig_list = []


def snake_segmentation_dots(
    path_to_file,
    path_to_origin,
    path_to_output
):
    multiple_frame_snakes = []
    multiple_frame_dots = []
    multiple_frame_distances = []
    multiple_frame_center_distances = []
    multiple_frame_outer_center_distances = []
    center_of_mass_list = []
    image = nib.load(path_to_file)
    try:
        image_origin = nib.load(path_to_origin)
    except:
        image_origin = None
    pixdim_xyz = image.header["pixdim"][1:4]
    pixel_area = pixdim_xyz[0] * pixdim_xyz[1]

    for frame in range(0, len(image.get_data()[:, :])):
        try:
            snakes = {}
            img = image.get_data()[:, :, frame]
            np_pixdata = np.array(img)
            np_pixdata = np.where(np_pixdata == SegmentationLevel.RV.value, 0, np_pixdata)
            an_array = np.where(np_pixdata < SegmentationLevel.LV.value, 0, np_pixdata)
            center_of_mass = ndimage.measurements.center_of_mass(an_array)
            print(center_of_mass)
            center_of_mass_list.append(center_of_mass)
            print("Applying Snakes")
            for seg_level in SegmentationLevel:
                if seg_level.value != SegmentationLevel.RV.value:
                    snakes[seg_level.value] = snake_seg(np_pixdata, center_of_mass, seg_level)
            multiple_frame_snakes.append(snakes)

            print("Applying Dots")
            dots = add_dots(pixel_area, center_of_mass, snakes)
            multiple_frame_dots.append(dots)
            distances = [d[2] for d in dots]  # Gets list of distances
            multiple_frame_distances.append(distances)
            center_distances = [d[3] for d in dots]
            outer_center_distances = [d[4] for d in dots]
            multiple_frame_center_distances.append(center_distances)
            multiple_frame_outer_center_distances.append(outer_center_distances)
        except Exception as e:
            fig_list.clear()
            print(e)
    print("Building Graphs")
    build_graphs(
        image_origin, image, multiple_frame_distances, multiple_frame_snakes, multiple_frame_dots,
        multiple_frame_center_distances, multiple_frame_outer_center_distances, center_of_mass_list
    )

    imageio.mimsave(path_to_output, fig_list)


def build_graphs(
    image_origin, image, multiple_frame_distances, multiple_frame_snakes, multiple_frame_dots,
    multiple_frame_center_distances, multiple_frame_outer_center_distances, center_of_mass_list
):
    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(3, 6, width_ratios=[2, 1, 1, 0, 1, 0], height_ratios=[2, 2, 2])
    ax = fig.add_subplot(gs[0:3, 0:2])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 4])
    ax4 = fig.add_subplot(gs[1, 2])
    ax5 = fig.add_subplot(gs[1, 4])
    ax9 = fig.add_subplot(gs[2, 2])
    ax11 = fig.add_subplot(gs[2, 4])
    ax6 = ax2.twinx()
    ax8 = ax4.twinx()
    ax10 = ax9.twinx()

    ax3.set_ylim(0, 20)
    ax5.set_ylim(0, 70)
    ax11.set_ylim(0, 70)
    ax3.set_title('thickness')
    ax5.set_title('center distance')
    ax11.set_title('outer center distance')

    custom_lines = []
    custom_lines2 = []
    ziped_frame_distances = list(zip(*multiple_frame_distances))
    ziped_center_distances = list(zip(*multiple_frame_center_distances))
    ziped_outer_center_distance = list(zip(*multiple_frame_outer_center_distances))
    for index, val in enumerate(ziped_frame_distances):
        dist = list(val)
        color = next(color_cycle)
        ax3.plot(dist, color=color)
        custom_lines.append(Line2D([0], [0], label=f"{round(min(dist), 2)} - {round(max(dist), 2)} mm",
                                   color=color, lw=4))
        custom_lines2.append(Line2D([0], [0], label='{:.2%}'.format(calculate_thickening(max(dist), min(dist))),
                                    color=color, lw=4))
    fig.legend(title='Thickness', handles=custom_lines, loc='upper right')
    fig.legend(title='Thickening', handles=custom_lines2, loc='lower right')
    custom_lines = []
    custom_lines2 = []
    for index, val in enumerate(zip(*multiple_frame_center_distances)):
        dist = list(val)
        color = next(color_cycle)
        ax5.plot(dist, color=color)
        custom_lines.append(Line2D([0], [0], label=f"{round(min(dist), 2)} - {round(max(dist), 2)} mm",
                                   color=color, lw=4))
        custom_lines2.append(Line2D([0], [0], label='{:.2%}'.format(calculate_thickening(max(dist), min(dist))),
                                    color=color, lw=4))
    fig.legend(title='Center Distance', handles=custom_lines, loc='upper left')
    fig.legend(title='Distance Percentage', handles=custom_lines2, loc='lower left')

    for index, val in enumerate(zip(*multiple_frame_outer_center_distances)):
        dist = list(val)
        color = next(color_cycle)
        ax11.plot(dist, color=color)
        custom_lines.append(Line2D([0], [0], label=f"{round(min(dist), 2)} - {round(max(dist), 2)} mm",
                                   color=color, lw=4))
        custom_lines2.append(Line2D([0], [0], label='{:.2%}'.format(calculate_thickening(max(dist), min(dist))),
                                    color=color, lw=4))

    for frame in range(len(multiple_frame_snakes)):
        ax2.set_ylim(0, 20)
        ax6.set_ylim(0, 1)
        ax8.set_ylim(0, 1)
        ax4.set_ylim(0, 60)
        ax9.set_ylim(0, 60)
        ax10.set_ylim(0, 1)
        ax2.set_title('thickness')
        ax4.set_title('center distance')
        ax9.set_title('outer center distance')
        if image_origin:
            img = image_origin.get_data()[:, :, frame]
        else:
            img = image.get_data()[:, :, frame]
        ax.imshow(img)
        # Print Raw Snakes
        snakes = multiple_frame_snakes[frame]
        ax.plot(center_of_mass_list[frame][1], center_of_mass_list[frame][0], 'o', color='r')
        for index in snakes:
            ax.plot(snakes[index][:, 1], snakes[index][:, 0], snake_colors[index], lw=3)
        dots = multiple_frame_dots[frame]
        # Print Dots
        count = 0
        for d, do, distance, center_distance, outer_center_distance in dots:
            color = next(color_cycle)
            ax.plot(d[0], d[1], 'o', color=color)
            ax.plot(do[0], do[1], 'o', color=color)
            ax2.plot(count, distance, 'o', color=color)
            ax9.plot(count, outer_center_distance, 'o', color=color)

            val = abs(min(ziped_frame_distances[count]) - distance) / min(ziped_frame_distances[count])
            ax6.plot(count, val, 'o', color='black')
            ax4.plot(count, center_distance, 'o', color=color)
            val = abs(max(ziped_center_distances[count]) - center_distance) / max(ziped_center_distances[count])
            ax8.plot(count, val, 'o', color='black')
            val = abs(max(ziped_outer_center_distance[count]) - outer_center_distance) / max(
                ziped_outer_center_distance[count])
            ax10.plot(count, val, 'o', color='black')

            count += 1

        fig.legend(title=f'{frame} Frame', loc='upper center')
        save_fig_to_list(fig)
        ax.clear()
        ax2.clear()
        ax4.clear()
        ax6.clear()
        ax8.clear()
        ax9.clear()
        ax10.clear()


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
                          max_iterations=snake_settings.max_iterations)


def add_dots(pixel_area, center_of_mass, snake_list):
    res = []

    inner_snake = snake_list[SegmentationLevel.LV.value]
    outer_snake = snake_list[SegmentationLevel.MYO.value]

    xn = outer_snake[:, 1]  # outer
    yn = outer_snake[:, 0]
    xn2 = inner_snake[:, 1]  # inner
    yn2 = inner_snake[:, 0]

    contour = LinearRing([(xn[j], yn[j]) for j in range(len(xn))])
    for i in range(len(xn2)):
        if i % POINT_INTERVAL == 0:
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
                center_distance = spatial_distance.euclidean(center_of_mass, (yn2[i], xn2[i]))
                print(center_distance, center_of_mass, (yn[i], xn[i]))
                outer_center_distance = spatial_distance.euclidean(center_of_mass, (yn[i], xn[i]))
                print(outer_center_distance, center_of_mass, (yn2[i], xn2[i]))
                res.append(
                    [(intersect.x, intersect.y), (xn2[i], yn2[i]), distance, center_distance, outer_center_distance])
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



#FOR TESTING
snake_segmentation_dots('/Users/antonioneto/Antonio/tese/Dados/OUTPUT_DIRECTORY_3D/Patient_CMD7_10.nii','/Users/antonioneto/Antonio/tese/Dados/Nifti/Patient_MCH1_11_0000.nii','/Users/antonioneto/Downloads/animation.gif')
import subprocess

try:
    cmd = "open -a 'google chrome' /Users/antonioneto/Downloads/animation.gif"
    subprocess.run(cmd, shell=True)
finally:
    cmd = "rm /Users/antonioneto/Downloads/animation.gif"
    subprocess.run(cmd, shell=True)
    print("Task finished")

