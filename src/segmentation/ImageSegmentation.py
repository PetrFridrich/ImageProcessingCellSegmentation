import matplotlib.pyplot as plt
import mahotas as mh
import numpy as np

import cv2

from PIL import ImageFilter, Image
from scipy import ndimage
 

class Segmenter():

    def __init__(self, image):

        # Image
        self.img_stage_0 = image

        # HyperParameters to reach the best segmentation
        self.weights = None


        
        return None
    

    def apply_segmentation(self):

        # Image sharpening
        self.img_sharpening()

        # Image conversion RGB -> grayscale
        self.img_RGB_to_gray()

        # Image conversion grayscale -> binary
        self.img_gray_to_bin()

        # Image erozion
        self.img_erode()        

        # Image dilatation
        self.img_dilate()

        # Image remove small regions in binary image (array)
        self.remove_small_regions_in_bin_img()
    
        # Image blur using Gauss kernel
        self.blur()



        return None
    

    """
    Here is implemenatation of Image Processing Methods!
    """

    def img_sharpening(self, img):

        RADIUS = 10
        PERCENT = 300
        THRESHOLD = 3

        img_pil = Image.fromarray(img, 'RGB')

        img = np.array(img_pil.filter(ImageFilter.UnsharpMask(radius = RADIUS, percent = PERCENT, threshold = THRESHOLD)))

        return img


    def img_RGB_to_gray(self, img):

        if self.weights is None:
            self.weights = [1/3, 1/3, 1/3]

        img = img[:,:,0]*self.weights[0] + img[:,:,1]*self.weights[1] + img[:,:,2]*self.weights[2]

        return img


    def img_gray_to_bin(self, img):

        if self.threshold_value is None: 
            self.threshold_value = self.img_stage_2.mean()

        img = img < self.threshold_value

        return img


    def img_erode(self, img):

        if self.num_of_morph_it is None:
            self.num_of_morph_it = 3
        
        if self.morph_mask is None:
            self.morph_mask = np.ones((3,3),np.uint8)

        img = cv2.erode(img, self.morph_mask, iterations=self.num_of_morph_it)

        return img


    def img_dilate(self, img):

        if self.num_of_morph_it is None:
            self.num_of_morph_it = 3
        
        if self.morph_mask is None:
            self.morph_mask = np.ones((3,3),np.uint8)

        img = cv2.dilate(img, self.morph_mask, iterations=self.num_of_morph_it)

        return img


    def convert_labeled_to_bin(self, img):

        if self.background is None:
            self.background = 0

        img = img != self.background

        return img


    def remove_small_regions_in_bin_img(self, img):

        if self.bin_region_min_size is None:
            self.bin_region_min_size = 200

        img, _ = mh.label(img)

        sizes = mh.labeled.labeled_size(img)

        img = mh.labeled.remove_regions_where(img, sizes < self.bin_region_min_size)
        
        img = self.convert_labeled_to_bin(img)

        return img


    def blur(self, img):

        if self.gauss_blur_sigma is None:
            self.gauss_blur_sigma = 10

        img = mh.gaussian_filter(img.astype(float), self.gauss_blur_sigma)

        # img = mh.stretch(img)

        return img
    
    def regional_maxima(self, img):

        

        return img





    def cell_correction(self, img):

        n_cells = np.amax(img)

        img_corrected = np.zeros_like(img)

        # Loop over cells in image
        for i in range(1,n_cells+1):

            # Image with only one cell
            img_single_cell = img == i

            # Cell is dilated (closing holes) and eroded (return to original size)
            img_single_cell = ndimage.grey_dilation(img_single_cell, size=(9, 9), mode='wrap')
            img_single_cell = ndimage.grey_erosion(img_single_cell, size=(9, 9), mode='wrap')

            # Assigning original label
            img_single_cell = img_single_cell * i

            # Inserting cell to final array (image)
            img_corrected = img_corrected + img_single_cell

            # Solution of potential cell overlap
            img_corrected[img_corrected > i] = i

        return img_corrected
    







if __name__ == '__main__':

    print('Hello, home!')