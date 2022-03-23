import imageio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage


def snake_segmentation_without_dots(path_to_file, patient_str, frame, snake_directory):
    nifti_image_pred = path_to_file
    print(nifti_image_pred)
    img_nifti = nib.load(nifti_image_pred)
    img = img_nifti.get_data()[:, :, frame]
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.gray()
    ax.imshow(img)
    np_pixdata = np.array(img)
    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    if an_array.shape[-1] == 3:
        img = rgb2gray(an_array)
    else:
        img = an_array
    s = np.linspace(0, 2 * np.pi, 400)
    r = 250 + 50 * np.sin(s)
    c = 280 + 50 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=0.015, beta=10, gamma=0.001, w_line=4, w_edge=8, max_iterations=5000)

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

    fig_name = snake_directory + '/' + str(patient_str) + '.png'
    fig.savefig(fig_name)
    return fig_name


def segment_patient(path_to_file, path_to_output_folder):
    fig_list = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    for i in range(30):
        try:
            fig_list.append(snake_segmentation_without_dots(path_to_file, patient_str, i, path_to_output_folder))
        except Exception as e:
            print(e)
    save_gif_2d(patient_str, fig_list)


def save_gif_2d(patient_str, fig_list):
    with imageio.get_writer(patient_str + '.gif', mode='I') as writer:
        for fig in fig_list:
            image = imageio.imread(fig)
            writer.append_data(image)
