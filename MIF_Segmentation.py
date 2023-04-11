"""
Class to handle creation and saving of MIF Segmentations
"""
import os
import re
import cv2
import numpy as np
import matplotlib.pyplot as plt

from utils import load_mif, get_mask

from skimage.morphology import black_tophat, skeletonize, convex_hull_image, area_closing, area_opening, binary_closing, disk  # noqa

class MIF_Segmentation:
    def __init__(self,
                 file_path,
                 num_core_biopsy,
                 channel_dict,
                 threshold_dict,
                 method_dict,
                 cache_dir,
                 save):

        self.file_path = file_path

        self.channel_dict = channel_dict

        self.masks = dict.fromkeys(channel_dict.keys(), None)

        self.thresh_dict = threshold_dict

        self.method_dict = method_dict

        # load mif image
        self.mif = load_mif(self.file_path)

        self.core_biopsy_mask = np.full_like(self.mif[0,:,:], False, dtype=bool)

        split_path = os.path.split(file_path)

        self.cache_dir = os.path.join(cache_dir, (split_path[-1]).split('.')[0])

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.num_core_biopsy = num_core_biopsy

        self.segmentations = {'inter': None,
                              'tubule': None}

        self.save = save

    def initialize(self):
        '''
        Function to create channel channel thresholds and create core biopsy mask
        if it does not exist. Otherwise, the core biopsy mask is loaded

        :return: None
        '''
        print('Initializing masks. This may take several minutes...')
        # get thresholds for the desired channels
        for channel, index in self.channel_dict.items():
            mask = get_mask(self.mif[index,:,:],
                            self.thresh_dict[channel][0],
                            self.thresh_dict[channel][1],
                            self.method_dict[channel])

            self.masks[channel] = mask

            self.core_biopsy_mask = np.logical_or(self.core_biopsy_mask, mask)
        print("Mask initialization finished. Creating Core Biopsy Mask")

        if not os.path.exists(os.path.join(self.cache_dir, 'core_biopsy.npy')):
            # create mask for each core biopsy
            # close holes
            self.core_biopsy_mask = area_closing(self.core_biopsy_mask, area_threshold=500**2, connectivity=2)

            # remove noise
            self.core_biopsy_mask= area_opening(self.core_biopsy_mask, area_threshold=5000, connectivity=2)

            print('Saving...')
            with open(os.path.join(self.cache_dir, 'core_biopsy.npy'), 'wb') as f:
                np.save(f, self.core_biopsy_mask)

        else:
            print('Loading ', os.path.join(self.cache_dir, 'core_biopsy.npy'))
            self.core_biopsy_mask = np.load(os.path.join(self.cache_dir, 'core_biopsy.npy'))


    def segment_interstism(self):
        """
        Function to segment and save the interstism mask
        :return: None
        """
        if not os.path.exists(os.path.join(self.cache_dir, 'inter_seg.npy')):

            # remove CD31 from collagen channel
            int_mask = np.logical_and(self.masks['inter'], np.logical_not(self.masks['cd31']))
            int_mask = np.logical_and(int_mask, np.logical_not(self.masks['cd34']))

            # remove all noise
            int_mask = area_opening(int_mask, area_threshold=50**2, connectivity=1)

            #close gaps
            int_mask = binary_closing(int_mask, disk(6))

            inter_seg = area_closing(int_mask, area_threshold=500, connectivity=1)


            self.segmentations['inter'] = inter_seg

            print('Saving...')
            with open(os.path.join(self.cache_dir, 'inter_seg.npy'), 'wb') as f:
                np.save(f, inter_seg)
        else:
            self.segmentations['inter'] = np.load(os.path.join(self.cache_dir, 'inter_seg.npy'))



    def segment_tubules(self):
        """
        Function to create and save tubule mask
        :return: None
        """
        percent_pixels = 0.5

        if type(self.segmentations['inter']) != np.ndarray:
            self.segment_interstism()

        # invert interstism mask
        int_mask = np.invert(self.segmentations['inter'])

        tubule_markers = ["aqp1", "aqp2", "ck7", "cal", "panck"]

        # create temporary mask for tubule markers
        # create final mask for tubule segmentation
        temp_mask = np.full_like(int_mask, False, dtype=bool)
        final_mask = np.full_like(int_mask, False, dtype=bool)
        # convert to type uint8 for cv2.drawContours
        final_mask.dtype = 'uint8'

        # add each tubule marker to temporary mask
        for channel, mask in self.masks.items():
            if channel in tubule_markers:
                temp_mask = np.logical_or(temp_mask, mask)

        # remove CD31 from tubules channel
        temp_mask = np.logical_and(temp_mask, np.logical_not(self.masks['cd31']))
        temp_mask = np.logical_and(temp_mask, np.logical_not(self.masks['cd34']))

        # draw core biopsy outline on interstism
        # this allows for tubules on edge of biopsy to be segmented
        all_signal = self.core_biopsy_mask.copy()
        all_signal.dtype = 'uint8'
        ext_contours, heirarchy = cv2.findContours(image=all_signal, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_NONE)
        ext_contours = list(ext_contours)
        ext_contours.sort(key=len, reverse=True)

        # draw external contours on interstism
        int_mask.dtype = 'uint8'
        for c in ext_contours[:self.num_core_biopsy]:
            cv2.drawContours(int_mask, [c], -1, 0, 3)

        contours, heirarchy = cv2.findContours(image=int_mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

        print("Contours: ", len(contours))
        for i, contour in enumerate(contours):

            test_mask = np.zeros_like(int_mask)
            cv2.drawContours(test_mask, [contour], 0, 255, -1)

            area = len(contour.ravel())

            # calculuate intenstity to draw tubule mask
            # this is to evaluate if proportion should be increased and
            # will be removed later
            this_prop = np.count_nonzero(np.logical_and(test_mask, temp_mask)) / area

            if np.count_nonzero(np.logical_and(test_mask,
                                               temp_mask)) / area > percent_pixels and area < 10000:  # use count_nonzero for performance gains
                cv2.drawContours(final_mask, [contour], 0, 255 * this_prop, -1)

            if i % 1000 == 0:
                print(i)

        print('Saving...')
        with open(os.path.join(self.cache_dir, 'tub_seg.npy'), 'wb') as f:
            np.save(f, final_mask)






