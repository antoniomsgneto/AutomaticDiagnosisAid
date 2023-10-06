import nibabel as nib
import os

# Specify the input 4D NIfTI file path and output directory
input_nifti_file = '/Users/antonioneto/Downloads/training/patient100/patient100_4d.nii.gz'
output_dir = '/Users/antonioneto/Downloads/training/patient100/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the 4D NIfTI file
nifti_img = nib.load(input_nifti_file)

# Get the 4D data array
data = nifti_img.get_fdata()

# Iterate through the slices
for slice_idx in range(data.shape[-2]):
    # Extract the 3D slice data
    slice_data = data[ :, :, slice_idx, :]

    # Create a new NIfTI image for the slice
    slice_img = nib.Nifti1Image(slice_data, affine=nifti_img.affine)

    # Define the output filename for the slice
    output_filename = os.path.join(output_dir, f'slice_{slice_idx+1}.nii.gz')

    # Save the slice as a 3D NIfTI file
    nib.save(slice_img, output_filename)

print(f"{data.shape[-2]} slices extracted and saved to '{output_dir}' directory.")
