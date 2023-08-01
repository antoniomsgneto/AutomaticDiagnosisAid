import datetime
from os import listdir
import sys

from segmentation.dot_segmentation_2 import snake_segmentation_dots

if __name__ == '__main__':
    if sys.argv[1]:
        for file in listdir(sys.argv[1]):
            if file.endswith('nii'):
                print(f'Processing Segmentation with Dots: {file}')
                try:
                    start_time = datetime.datetime.now()
                    snake_segmentation_dots(
                        sys.argv[1] + file, # Path to each file to process
                        sys.argv[2] + file.replace('.nii','_0000.nii'), # Path to original file .nii file
                        sys.argv[3] + file[:-3] + '.gif' # Path to output gif
                    )
                    end_time = datetime.datetime.now()
                    print(f'Processing time for {file} : {end_time-start_time}')
                except Exception as e:
                    print(f'Error processing {file} : {e}')
