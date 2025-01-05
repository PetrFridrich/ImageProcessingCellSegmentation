import matplotlib.pyplot as plt
import mahotas as mh
import numpy as np
from pathlib import Path
from matplotlib import cm

import cv2

from PIL import ImageFilter, Image
from scipy import ndimage
 

class Segmenter():

    def __init__(self, image:np.ndarray, 
                       sharpening_radius:int=10,
                       sharpening_percent:int=300,
                       sharpening_threshold:int=3,
                       weights:np.ndarray=[0.05, 0.9, 0.05],
                       threshold_value:int=None,
                       morph_mask:np.ndarray=np.ones((3,3),np.uint8),
                       num_of_morph_it:int=3,
                       bin_region_min_size:int=200,
                       labeled_region_min_size:int=200,
                       background:int=0,
                       gauss_blur_sigma:int=10):

        # Image
        self.img_stage_c0 = image

        # HyperParameters to reach the best segmentation
        # Sharpening
        self.sharpening_radius = sharpening_radius
        self.sharpening_percent = sharpening_percent
        self.sharpening_threshold = sharpening_threshold

        # RGB -> grayscale
        self.weights = weights

        # grayscale -> bin
        self.threshold_value = threshold_value

        # minimum region size
        self.bin_region_min_size = bin_region_min_size
        self.labeled_region_min_size = labeled_region_min_size

        # label considered as background 
        self.background = background

        # Morphological filter parameters
        self.morph_mask = morph_mask
        self.num_of_morph_it = num_of_morph_it

        # Gauss blur sigma
        self.gauss_blur_sigma = gauss_blur_sigma
        
        return None
    

    def apply_segmentation(self):
        """
        Here is implementation of Image Processing Methods!
        Every single method is one small step in segmentation process.
        For vizual interpretation look at ./docs/README.md section Pipeline
        """

        # ----------------------------------------------------- Center ----------------------------------------------------- #
        # Image sharpening
        self.image_stage_c1 = self.img_sharpening(img=self.img_stage_c0)

        # Image conversion RGB -> grayscale
        self.image_stage_c2 = self.img_RGB_to_gray(img=self.image_stage_c1)

        # ----------------------------------------------------- Left ----------------------------------------------------- #
        # Image conversion grayscale -> binary
        self.image_stage_l1 = self.img_gray_to_bin(img=self.image_stage_c2)
     
        # Image erozion
        self.image_stage_l2 = self.img_erode(img=self.image_stage_l1)

        # Image dilatation
        self.image_stage_l3 = self.img_dilate(img=self.image_stage_l2)

        # Remove small regions in binary image (array)
        self.image_stage_l4 = self.remove_small_regions_in_bin_img(img=self.image_stage_l3)

        # Reverse distance map in binary image
        self.image_stage_l5 = self.distance_transform_in_bin_img(self.image_stage_l4)

        # ----------------------------------------------------- Right ----------------------------------------------------- #
        # Image blur using Gauss kernel
        self.image_stage_r1 = self.blur(img=self.image_stage_c2)

        # Localization of regional maxima in grayscale image (array)
        self.image_stage_r2 = self.regional_maxima(self.image_stage_r1)

        # Label each regional maxima
        self.image_stage_r3 = self.label_img(self.image_stage_r2)

        # Watershed segmentation (surface = reverse_distance_map, markers = labeled_regional_maxima)
        self.image_stage_r4 = self.apply_watershed_segmentation(surface=self.image_stage_l5, markers=self.image_stage_r3)

        # ----------------------------------------------------- Center ----------------------------------------------------- #
        # Application of binary mask to watershed segmented image
        self.image_stage_c3 = self.apply_binary_mask(self.image_stage_r4, self.image_stage_l4)

        # Remove small regions in labeled image (array)
        self.image_stage_c4 = self.remove_small_regions_in_labeled_img(self.image_stage_c3)

        # Relabel image after removing regions
        self.image_stage_c5 = self.relabel_img(self.image_stage_c4)

        # Repairing cells in image using morphological operations
        self.image_stage_c6 = self.cell_correction(self.image_stage_c5)

        # Removing regions touching image borders
        self.image_stage_c7 = self.remove_bordering_regions(self.image_stage_c6)

        # Relabel image after removing bordering regions
        self.image_stage_c8 = self.relabel_img(self.image_stage_c7)

        return None
    

    def save_image_flow(self, output_path:Path):

        # Output path
        output_path.mkdir(parents=True, exist_ok=True)

        # Color map settings
        cm_jet = cm.get_cmap('jet')
        cm_jet.set_under('k')

        # Image saving
        # 
        plt.imsave(output_path / "000_Input_Image.png", self.img_stage_c0)

        # 
        plt.imsave(output_path / "001_Sharpened_Image.png", self.image_stage_c1)

        # 
        plt.imsave(output_path / "002_Grayscale_Image.png", self.image_stage_c2, cmap='gray')

        # 
        plt.imsave(output_path / "003_Binary_Image.png", self.image_stage_l1, cmap='gray')

        # 
        plt.imsave(output_path / "004_Binary_Eroded_Image.png", self.image_stage_l2, cmap='gray')

        # 
        plt.imsave(output_path / "005_Binary_Dilated_Image.png", self.image_stage_l3, cmap='gray')

        # 
        plt.imsave(output_path / "006_Binary_Removed_Small_Regions_Image.png", self.image_stage_l4, cmap='gray')

        # 
        plt.imsave(output_path / "007_Distance_Transform_Image.png", self.image_stage_l5, cmap='gray')

        # 
        plt.imsave(output_path / "008_Blur_Grayscale_Image.png", self.image_stage_r1, cmap='gray')

        # 
        plt.imsave(output_path / "009_Regional_Maxima_Image.png", self.image_stage_r2, cmap='gray')

        # 
        plt.imsave(output_path / "010_Regional_Maxima_Labeled_Image.png", self.image_stage_r3, cmap=cm_jet, vmin=1)

        #
        plt.imsave(output_path / "011_Applied_Watershed.png", self.image_stage_r4, cmap=cm_jet)

        #
        plt.imsave(output_path / "012_Segmented_Image.png", self.image_stage_c3, cmap=cm_jet, vmin=1)

        #
        plt.imsave(output_path / "013_Segmented_Removed_Small_Regions_Image.png", self.image_stage_c4, cmap=cm_jet, vmin=1)

        #
        plt.imsave(output_path / "014_Segmented_Relabeled_Image.png", self.image_stage_c5, cmap=cm_jet, vmin=1)

        #
        plt.imsave(output_path / "015_Segmented_Cell_Correction_Image.png", self.image_stage_c6, cmap=cm_jet, vmin=1)

        #
        plt.imsave(output_path / "016_Segmented_Removed_Bordering_Image.png", self.image_stage_c7, cmap=cm_jet, vmin=1)

        #
        plt.imsave(output_path / "017_Segmented_Relabeled_Image.png", self.image_stage_c8, cmap=cm_jet, vmin=1)

        return None

    """
    Every method used in segmentation process.
    """
    def img_sharpening(self, img:np.ndarray):

        img_pil = Image.fromarray(img, 'RGB')

        img = np.array(img_pil.filter(ImageFilter.UnsharpMask(radius=self.sharpening_radius, 
                                                              percent=self.sharpening_percent, 
                                                              threshold=self.sharpening_threshold)))

        return img


    def img_RGB_to_gray(self, img:np.ndarray):

        img = img[:,:,0]*self.weights[0] + img[:,:,1]*self.weights[1] + img[:,:,2]*self.weights[2]

        return img


    def img_gray_to_bin(self, img:np.ndarray):

        if self.threshold_value is None: 
            self.threshold_value = img.mean()

        img = img > self.threshold_value

        return img.astype(np.uint8)


    def img_erode(self, img:np.ndarray):

        img = cv2.erode(img, self.morph_mask, iterations=self.num_of_morph_it)

        return img


    def img_dilate(self, img:np.ndarray):

        img = cv2.dilate(img, self.morph_mask, iterations=self.num_of_morph_it)

        return img


    def convert_labeled_to_bin(self, img:np.ndarray):

        img = img != self.background

        return img


    def remove_small_regions_in_bin_img(self, img:np.ndarray):

        img, _ = mh.label(img)

        sizes = mh.labeled.labeled_size(img)

        img = mh.labeled.remove_regions_where(img, sizes < self.bin_region_min_size)
        
        img = self.convert_labeled_to_bin(img)

        return img


    def remove_small_regions_in_labeled_img(self, img:np.ndarray):

        sizes = mh.labeled.labeled_size(img)

        img = mh.labeled.remove_regions_where(img, sizes < self.labeled_region_min_size)

        return img


    def remove_bordering_regions(self, img:np.ndarray):

        img = mh.labeled.remove_bordering(img)

        return img


    def blur(self, img:np.ndarray):

        img = mh.gaussian_filter(img.astype(float), self.gauss_blur_sigma)

        img = mh.stretch(img)

        return img
    

    def regional_maxima(self, img:np.ndarray):

        img = mh.regmax(img)

        return img
    

    def distance_transform_in_bin_img(self, img:np.ndarray):

        img = mh.distance(img)

        img = 255 - mh.stretch(img)        

        return img

    
    def apply_watershed_segmentation(self, surface:np.ndarray, markers:np.ndarray):

        img = mh.cwatershed(surface, markers)

        return img


    def apply_binary_mask(self, watershed:np.ndarray, bin_img:np.ndarray):

        img = watershed * bin_img

        return img


    def label_img(self, img:np.ndarray):

        img, _ = mh.label(img)

        return img


    def relabel_img(self, img:np.ndarray):

        img, _ = mh.labeled.relabel(img)

        return img


    def cell_correction(self, img:np.ndarray):

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