from moviepy.editor import VideoFileClip
import sys

filename = '/Users/antonioneto/Downloads/mygraph (2).gif'

# Load the MP4 video and play it
try:
    clip = VideoFileClip(filename)
    # Play the video in a loop
    while True:
        try:
            clip.preview()
        except KeyboardInterrupt:
            # If the user interrupts the loop (e.g. by pressing Ctrl+C), exit the loop
            break

    # Clean up the resources used by the video clip
    clip.close()
except Exception as e:
    print(f'Error displaying {filename}: {e}')

