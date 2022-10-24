import imageio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import active_contour
from skimage.color import rgb2gray
from skimage.filters import gaussian
from scipy import ndimage
from skimage.draw import polygon, polygon_perimeter

from utilities.dicom2nifticonverter import save_array_list_to_nifti


def snake_segmentation_without_dots(path_to_file, patient_str, frame, snake_directory):
    nifti_image_pred = path_to_file
    print(nifti_image_pred)
    img_nifti = nib.load(nifti_image_pred)
    img = img_nifti.get_data()[:, :, frame]
    print(img)
    print(img.shape)
    img_shape = img.shape
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111)
    plt.gray()
    #ax.imshow(img)
    np_pixdata = np.array(img)
    # np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 2)])
    an_array = np.where(np_pixdata != 2, 0, np_pixdata)
    img = rgb2gray(an_array)

    s = np.linspace(0, 2 * np.pi, 500)
    r = 250 + 50 * np.sin(s)
    c = 280 + 50 * np.cos(s)
    init = np.array([r, c]).T
    snake = active_contour(gaussian(img, 3),
                           init, boundary_condition="periodic",
                           alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)

    #np_counter = np.count_nonzero(np_pixdata[(np_pixdata == 3)])
    an_array = np.where(np_pixdata != 3, 0, np_pixdata)
    center_of_mass = ndimage.measurements.center_of_mass(an_array)
    img = rgb2gray(an_array)
    snake2 = active_contour(gaussian(img, 3),
                            init, boundary_condition="periodic",
                            alpha=0.015, beta=10, gamma=0.001, w_line=-0.2, w_edge=4, max_iterations=5000)
    snake_array = np.zeros([img_shape[0]*100, img_shape[1]*100])
    points_r, points_c = list(map(lambda x: round(x*100), snake[:, 0])), list(map(lambda x: round(x*100), snake[:, 1]))
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    snake_array[perimeter_r, perimeter_c] = 1
    snake_array[points_r, points_c] = 1
    snake_array[interior_r, interior_c] = 1
    points_r, points_c = list(map(lambda x: round(x*100), snake2[:, 0])), list(
        map(lambda x: round(x*100), snake2[:, 1]))
    interior_r, interior_c = polygon(points_r, points_c)
    perimeter_r, perimeter_c = polygon_perimeter(points_r, points_c)
    snake_array[perimeter_r, perimeter_c] = 2
    snake_array[points_r, points_c] = 2
    snake_array[interior_r, interior_c] = 2

    print(snake_array)
    final_snake_array = snake_array[::100, ::100]
    # for x,y in snake2:
   #ax.imshow(snake_array[::100,::100])
    #for x,y in snake2:
        #snake_array[round(x*100), round(y*100)] = 3
    #make_region(snake_array)
    #snake_array = make_region(snake_array[::3])
    #snake_array = rgb2gray(snake_array)
    #ax.imshow(snake_array)
    return snake_array[::100,::100]
    #ax.plot(list(map(lambda x: round(x * 100), (snake[:, 1]))),
     #       list(map(lambda x: round(x * 100), (snake[:, 0]))), '-r', lw=3)
    #ax.plot(list(map(lambda x: round(x * 100), (snake2[:, 1]))),
   #         list(map(lambda x: round(x * 100), (snake2[:, 0]))), '-b', lw=3)
    #ax.plot(center_of_mass[1], center_of_mass[0], "or")
    #ax.axis([0, img.shape[1], img.shape[0], 0])

    #fig_name = snake_directory + '/' + str(patient_str)+str(frame) + '.png'
    #fig.savefig(fig_name)
    #return fig_name


def segment_patient(path_to_file, path_to_output_folder):
    fig_list = []
    patient_str = path_to_file.split('.')[0].split('/')[-1]
    for i in range(30):
        try:
            snake = snake_segmentation_without_dots(path_to_file, patient_str, i, path_to_output_folder)
            fig_list.append(snake)
        except Exception as e:
            print(e)
    #print(fig_list[0].shape)
    save_array_list_to_nifti(fig_list,fig_list[0].shape[0],fig_list[0].shape[1],path_to_output_folder+'/'+patient_str)

def make_region(snake):
    # data = scipy.misc.imread(snake)
    data = snake
    assert data.ndim == 2, "Image must be monochromatic"

    # finds and number all disjoint white regions of the image
    is_white = data == 0
    # print(is_white)
    labels, n = ndimage.measurements.label(is_white)
    # print(labels)
    # print(n)
    # get a set of all the region ids which are on the edge - we should not fill these
    on_border = set(labels[:, 0]) | set(labels[:, -1]) | set(labels[0, :]) | set(labels[-1, :])
    # print(on_border)
    for label in range(1, n + 1):  # label 0 is all the black pixels
        if label not in on_border:
            # turn every pixel with that label to black
            print("HERE")
            data[labels == label] = 1
    return data


def save_gif_2d(patient_str, fig_list):
    with imageio.get_writer(patient_str + '.gif', mode='I') as writer:
        for fig in fig_list:
            image = imageio.imread(fig)
            writer.append_data(image)


segment_patient('/Users/antonioneto/Antonio/tese/Dados/test/Patient_0_1.nii.gz','/Users/antonioneto/Antonio/tese/Dados/SnakesResult')