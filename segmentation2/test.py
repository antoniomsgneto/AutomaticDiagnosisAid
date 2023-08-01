import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, measure
from skimage.segmentation import (morphological_chan_vese,
                                  morphological_geodesic_active_contour,
                                  inverse_gaussian_gradient,
                                  checkerboard_level_set)

import nibabel as nib

def store_evolution_in(lst):
    """Returns a callback function to store the evolution of the level sets in
    the given list.
    """

    def _store(x):
        lst.append(np.copy(x))

    return _store


# Morphological ACWE
#image = img_as_float(data.camera())

img_path = '/Users/antonioneto/Antonio/tese/Dados/OUTPUT_DIRECTORY_3D/Patient_MCH1_11.nii'
frame = 1
image = nib.load(img_path)
img = image.get_data()[:, :, frame]
np_pixdata = np.array(img)
an_array = np.where(np_pixdata != 2, 0, np_pixdata)
image = an_array

# Initial level set
#init_ls = checkerboard_level_set(image.shape, 6)
init_mask = np.zeros(an_array.shape, dtype=bool)
init_ls = init_mask[:200, :200] = 2
# List with intermediate results for plotting the evolution
evolution = []
callback = store_evolution_in(evolution)
ls = morphological_chan_vese(image, iterations=10, init_level_set=init_ls,
                             smoothing=3, iter_callback=callback)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.flatten()

ax[0].imshow(image, cmap="gray")
ax[0].set_axis_off()
ax[0].contour(ls, [0.5], colors='r')
ax[0].set_title("Morphological ACWE segmentation", fontsize=12)

ax[1].imshow(ls, cmap="gray")
ax[1].set_axis_off()
contour = ax[1].contour(evolution[2], [0.5], colors='g')
contour.collections[0].set_label("Iteration 2")
contour = ax[1].contour(evolution[7], [0.5], colors='y')
contour.collections[0].set_label("Iteration 7")
contour = ax[1].contour(evolution[-1], [0.5], colors='r')
contour.collections[0].set_label("Iteration 35")
ax[1].legend(loc="upper right")
title = "Morphological ACWE evolution"
ax[1].set_title(title, fontsize=12)

contours = measure.find_contours(ls, 0.5)
res = []



ax[2].imshow(image, cmap="gray")
for i in contours:
    if len(contours) > 100:
    #ax[2].contour(contours[index], [0.5], colors='r')
        ax[2].plot(i[:, 1], i[:, 0], 'r', lw=3)


ax[2].set_axis_off()
ax[2].contour(ls, [0.5], colors='r')
ax[2].set_title("Morphological GAC segmentation", fontsize=12)

ax[3].imshow(ls, cmap="gray")
ax[3].set_axis_off()

ax[3].legend(loc="upper right")
title = "Morphological GAC evolution"
ax[3].set_title(title, fontsize=12)

fig.tight_layout()
plt.show()

"""
for i in range(0,len(contours)):
    morph = contours[i]
    contour_points = []
    for y in range(0, len(morph)):
        for x in range(0, len(morph[y])):
            if morph[x, y] >= 1:
                contour_points.append([x, y])
    res.append(contour_points)
"""