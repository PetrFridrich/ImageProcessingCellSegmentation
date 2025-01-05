import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import mahotas as mh
from skimage import io

from segmentation.ImageSegmentation import Segmenter

def main():

    IMAGE_PATH = Path(".", "data", "img_1.tif")
    OUTPUT_PATH = Path(".", "results", IMAGE_PATH.stem, "IMG_flow")

    img = io.imread(IMAGE_PATH)

    img_segmenter = Segmenter(image=img)
    
    img_segmenter.apply_segmentation()
    img_segmenter.save_image_flow(OUTPUT_PATH)






    return None

if __name__ == '__main__':
    print('Hello, world!')
    main()