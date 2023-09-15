"""
Class to handle creation and saving of MIF Segmentations
"""
import re
import os
# import sys

from skimage.segmentation import mark_boundaries
from skimage import segmentation
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
import pandas as pd
import cv2
from openslide import open_slide
from joblib import load

# sys.path.append(
#     "/home/alpaca/anaconda3/envs/smith_comp_path/lib/python3.10/site-packages/")


from skimage.morphology import black_tophat, skeletonize, convex_hull_image, area_closing, area_opening, binary_closing, binary_dilation, disk, diamond  # noqa
from utils import load_mif, get_mask, get_mask_metrics, plot_mask
from metric_utils import matched_detection_metrics, pixelwise_metrics

from yatt.WSI import WSI
from yatt.read_image import get_patch_at_mpp
from wsi_reg.load_aligned_patches import get_src_patch_aligned_to_targ_bbox,\
    get_src_bbox_aligned_to_targ_patch


class MIF_Segmentation:
    """"
    Class to handle MIF Segmentations 

    """

    def __init__(self,
                 file_path: str,
                 num_core_biopsy: int,
                 channel_dict: list,
                 threshold_dict: dict,
                 method_dict: dict,
                 cache_dir: str,
                 frozen: bool = False,
                 patch: list = None,
                 save: bool = False) -> None:

        self.file_path = file_path

        self.channel_dict = channel_dict

        self.masks = dict.fromkeys(channel_dict.keys(), None)

        self.thresh_dict = threshold_dict

        self.method_dict = method_dict

        self.frozen = frozen

        self.patch = patch

        # load mif image
        self.mif = load_mif(self.file_path)

        if patch:
            self.patch = patch
            self.mif = self.mif[:, patch[0]:patch[1], patch[2]:patch[3]]

        self.core_biopsy_mask = np.full_like(
            self.mif[0, :, :], False, dtype=bool)

        split_path = os.path.split(file_path)

        self.cache_dir = os.path.join(
            cache_dir, (split_path[-1]).split('.')[0] + '-skeleton')

        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.num_core_biopsy = num_core_biopsy

        self.segmentations = {'inter': None,
                              'tubule': None,
                              'glom_mask': None,
                              'vessels': None}

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
            mask = get_mask(self.mif[index, :, :],
                            self.thresh_dict[channel][0],
                            self.thresh_dict[channel][1],
                            self.method_dict[channel])

            self.masks[channel] = mask

            self.core_biopsy_mask = np.logical_or(self.core_biopsy_mask, mask)
        print("Mask initialization finished. Creating Core Biopsy Mask")

        if not os.path.exists(os.path.join(self.cache_dir, 'core_biopsy.npy')):
            # create mask for each core biopsy
            # close holes
            self.core_biopsy_mask = area_closing(
                self.core_biopsy_mask, area_threshold=500**2, connectivity=2)

            # remove noise
            self.core_biopsy_mask = area_opening(
                self.core_biopsy_mask, area_threshold=5000, connectivity=2)

            print('Saving...')
            with open(os.path.join(self.cache_dir, 'core_biopsy.npy'), 'wb') as f:
                np.save(f, self.core_biopsy_mask)

            if self.patch:
                self.core_biopsy_mask = self.core_biopsy_mask[
                    self.patch[0]:self.patch[1], self.patch[2]:self.patch[3]]

        else:
            print('Loading ', os.path.join(self.cache_dir, 'core_biopsy.npy'))
            self.core_biopsy_mask = np.load(
                os.path.join(self.cache_dir, 'core_biopsy.npy'))

            if type(self.patch) is list:
                self.core_biopsy_mask = self.core_biopsy_mask[
                    self.patch[0]:self.patch[1], self.patch[2]:self.patch[3]]

    def segment_vessels(self):
        """_summary_: Function to segment the vessels using the SMA channel
        """
        if not os.path.exists(os.path.join(self.cache_dir, 'vessels.npy')):
            # remove vessels

            sma_mask = self.masks['sma']

            sma_mask.dtype = 'uint8'

            # fill in all vessels

            vessels, _ = cv2.findContours(
                sma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cv2.drawContours(sma_mask, vessels, -1, 255, -1)

            self.segmentations['vessels'] = sma_mask

            # print('Saving...')
            with open(os.path.join(self.cache_dir, 'vessels.npy'), 'wb') as f:
                np.save(f, sma_mask)

        else:
            print('Loading ', os.path.join(self.cache_dir, 'vessels.npy'))
            self.segmentations['vessels'] = np.load(
                os.path.join(self.cache_dir, 'vessels.npy'))

    def initialize_glomeruli(self, channel: str = 'cd34'):
        """
        This function is the first path to extracting the glomeruli. It finds the most
        circular sections from the interstism mask that have CD34 present. 

        :return:
        """
        if not os.path.exists(os.path.join(self.cache_dir, 'initial_glom.npy')):

            tub_masks = ["aqp1", "aqp2", "ck7", "cal", "panck"]

            percent_pixels = 0.0

            # preprocess CD31 channel
            # create glomeruli mask
            glom_mask = np.copy(self.masks[channel])

            # remove vessels
            if type(self.segmentations['vessels']) != np.ndarray:
                self.segment_vessels()

            sma_mask = self.segmentations['vessels']

            glom_mask = np.logical_and(glom_mask, np.logical_not(sma_mask))

            # close holes
            # glom_mask = binary_closing(glom_mask, disk(8))

            # remove all small regions from mask
            glom_mask = binary_closing(glom_mask, disk(3))
            glom_mask = area_opening(glom_mask, area_threshold=1000)

            # find all holes in not in interstism and tubule masks
            int_mask = self.masks['inter']

            for mask in tub_masks:
                int_mask = np.logical_or(int_mask, self.masks[mask])
            int_mask = np.logical_and(int_mask, np.logical_not(glom_mask))

            int_mask = area_opening(int_mask, area_threshold=5000)
            int_mask = binary_closing(int_mask, disk(6))

            # remove small open areas created by capillaries
            # 10000 was chosen as it is slightly smaller than the
            # area of the smallest glomeruli
            int_mask = area_closing(int_mask, area_threshold=10000)

            int_mask = np.invert(int_mask)
            int_mask.dtype = 'uint8'

            # draw outline of core biopsies on mask
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

            plt.imsave('int_mask_for_glom.png', int_mask)

            # get contours
            contours, hierarchy = cv2.findContours(int_mask, mode=cv2.RETR_LIST,
                                                   method=cv2.CHAIN_APPROX_NONE)

            final_mask = np.zeros_like(int_mask)
            print('contours: ', len(contours))
            for i, cont in enumerate(contours):

                test_mask = np.zeros_like(final_mask)
                test_mask.dtype = 'uint8'
                cv2.drawContours(test_mask, [cont], 0, 255, -1)

                area = np.count_nonzero(test_mask)

                if (np.count_nonzero(np.logical_and(test_mask,
                                                    glom_mask)) / area) > percent_pixels and area < 3000000:
                    # check for circularity:
                    (x, y), radius = cv2.minEnclosingCircle(cont)

                    circ_area = np.pi * radius ** 2

                    print('ratio is ', area / circ_area)
                    if area / circ_area < 1.5 and area / circ_area > 0.3:
                        print('area is ', area)
                        cv2.drawContours(final_mask, [cont], 0, 255, -1)
                    # cv2.drawContours(final_mask, [cont], 0, 255, -1)

            self.segmentations['glom_mask'] = final_mask

            # print('Saving...')
            with open(os.path.join(self.cache_dir, 'initial_glom.npy'), 'wb') as f:
                np.save(f, final_mask)

        else:
            print('Loading ', os.path.join(self.cache_dir, 'initial_glom.npy'))
            self.segmentations['glom_mask'] = np.load(
                os.path.join(self.cache_dir, 'initial_glom.npy'))
            self.segmentations['vessels'] = np.load(
                os.path.join(self.cache_dir, 'vessels.npy'))

    def segment_interstism(self):
        """
        Function to segment and save the interstism mask
        :return: None
        """
        if not os.path.exists(os.path.join(self.cache_dir, 'inter_seg.npy')):

            # check for glomeruli mask
            if type(self.segmentations['glom_mask']) != np.ndarray:
                self.initialize_glomeruli(channel='cd34')

            else:
                glom_mask = self.segmentations['glom_mask']

            # remove glomeruli from collagen channel
            int_mask = np.logical_and(
                self.masks['inter'], np.invert(glom_mask))

            # remove vessels
            int_mask = np.logical_and(
                int_mask, np.invert(self.segmentations['vessels']))

            int_closed = binary_closing(int_mask, disk(10))
            # int_closed = binary_closing(int_mask, disk(15))
            # int_skeleton = skeletonize(int_closed)
            # int_mask = np.logical_or(int_mask, int_skeleton)
            # plt.imsave('tests/skeleton.png', int_skeleton)
            # footprint = np.arange(30//2)
            # footprint = footprint * 1 | footprint[:, None]
            # int_mask = binary_closing(int_mask, footprint)

            # remove all noise
            int_mask = area_opening(
                int_mask, area_threshold=50**2, connectivity=1)

            inter_seg = area_closing(
                int_mask, area_threshold=500, connectivity=1)

            self.segmentations['inter'] = inter_seg

            print('Saving...')
            with open(os.path.join(self.cache_dir, 'inter_seg.npy'), 'wb') as f:
                np.save(f, inter_seg)
        else:
            self.segmentations['inter'] = np.load(
                os.path.join(self.cache_dir, 'inter_seg.npy'))

    def segment_interstism_frozen(self):
        """
        Function to segment and save the interstism mask from a frozen biopsy
        :return: None
        """
        if not os.path.exists(os.path.join(self.cache_dir, 'inter_seg.npy')):

            # remove CD31 from collagen channel
            int_mask = self.masks['inter']

            # remove all noise
            int_mask = area_opening(
                int_mask, area_threshold=50**2, connectivity=1)

            # close gaps
            int_mask = binary_closing(int_mask, disk(6))

            inter_seg = area_closing(
                int_mask, area_threshold=500, connectivity=1)

            self.segmentations['inter'] = inter_seg

            print('Saving...')
            with open(os.path.join(self.cache_dir, 'inter_seg.npy'), 'wb') as f:
                np.save(f, inter_seg)
        else:
            if self.patch:
                self.segmentations['inter'] = np.load(os.path.join(self.cache_dir, 'inter_seg.npy'))[
                    self.patch[0]:self.patch[1], self.patch[2]:self.patch[3]]
            else:
                self.segmentations['inter'] = np.load(
                    os.path.join(self.cache_dir, 'inter_seg.npy'))

    def segment_tubules_frozen(self):
        """_summary_ Function to segmenet the tubules from a frozen biopsy
        """
        if type(self.segmentations['inter']) != np.ndarray:
            self.segment_interstism()

        int_mask = self.segmentations['inter']
        # plt.imsave('tests/im.png', int_mask)

        tubule_markers = ["aqp1", "aqp2", "ck7", "cal", "panck"]

        temp_mask = np.full_like(int_mask, False, dtype=bool)
        final_mask = np.full_like(int_mask, False, dtype=bool)
        # convert to type uint8 for cv2.drawContours
        final_mask.dtype = 'uint8'

        # add each tubule marker to temporary mask
        for channel, mask in self.masks.items():
            if channel in tubule_markers:
                temp_mask = np.logical_or(temp_mask, mask)

        # remove interstism
        temp_mask = np.logical_and(temp_mask, np.invert(int_mask))

        temp_mask = area_closing(temp_mask, 100)

        # plt.figure()
        # plt.imshow(temp_mask)
        # plt.imsave('tests/temp.png', temp_mask)

        # felzenshwalb
        felz = segmentation.felzenszwalb(temp_mask, scale=10, min_size=100)
        print('got felz')
        plt.figure()
        plt.imshow(felz)
        plt.imsave('tests/felz.png', felz)
        print('saving boundaries')
        plt.imsave('tests/bound.png', mark_boundaries(int_mask, felz))

        # cv= segmentation.morphological_chan_vese(temp_mask, num_iter=235)
        #
        # plt.imsave('tests/cv.png', mark_boundaries(int_mask, cv))
        #
        # cv2 = segmentation.morphological_chan_vese(temp_mask, num_iter=235, init_level_set = 'disk')
        # plt.imsave('tests/cv2.png', mark_boundaries(int_mask, cv2))

    def segment_capillaries(self, channel: str = 'cd31'):
        """_summary_: Function to segment capillaries 

        Args:
            channel (str, optional): Channel to segment capillaries from. Defaults to 'cd31'.
        """

        capillary_mask = np.logical_and(
            self.masks[channel], np.invert(self.segmentations['glom_mask']))
        capillary_mask = np.logical_and(
            capillary_mask, np.invert(self.segmentations["vessels"]))

        with open(os.path.join(self.cache_dir, 'capillary_seg.npy'), 'wb') as f:
            np.save(f, capillary_mask)

    def segment_tubules(self):
        """
        Function to create and save tubule mask
        :return: None
        """
        if not os.path.exists(os.path.join(self.cache_dir, 'tub_seg.npy')):

            percent_pixels_aqp1 = 0.2
            percent_pixels_all = 0.1

            if type(self.segmentations['glom_mask']) != np.ndarray:
                self.initialize_glomeruli(channel='cd34')

            if type(self.segmentations['inter']) != np.ndarray:
                self.segment_interstism()

            # remove vessels
            if type(self.segmentations['vessels']) != np.ndarray:
                self.segment_vessels()

            # initialize glom mask
            glom_mask = self.segmentations['glom_mask']

            # invert interstism mask
            int_mask = np.invert(self.segmentations['inter'])

            tubule_markers = ["aqp2", "ck7", "cal", "panck"]

            aqp1_mask = self.masks["aqp1"]

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

            # remove glomeruli from tubules channel
            temp_mask = np.logical_and(temp_mask, np.invert(glom_mask))

            # remove vessels from tubules
            temp_mask = np.logical_and(
                temp_mask, np.invert(self.segmentations['vessels']))

            # draw core biopsy outline on interstism
            # this allows for tubules on edge of biopsy to be segmented
            all_signal = self.core_biopsy_mask.copy()

            # remove exterior signal
            int_mask = np.logical_and(int_mask, (all_signal))
            int_mask.dtype = 'uint8'

            contours, heirarchy = cv2.findContours(
                image=int_mask, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)

            print("Contours: ", len(contours))
            count_draw = 0
            for i, contour in enumerate(contours):

                test_mask = np.zeros_like(int_mask)
                cv2.drawContours(test_mask, [contour], 0, 255, -1)

                area = np.count_nonzero(test_mask)

                # calculuate intenstity to draw tubule mask
                # this is to evaluate if proportion should be increased and
                # will be removed later
                prop_all_markers = np.count_nonzero(
                    np.logical_and(test_mask, temp_mask)) / area

                prop_aqp1 = np.count_nonzero(
                    np.logical_and(test_mask, aqp1_mask)) / area

                if prop_all_markers > percent_pixels_all or prop_aqp1 > percent_pixels_aqp1:

                    count_draw += 1
                    cv2.drawContours(final_mask, [contour], 0, 255, -1)

                if i % 1000 == 0:
                    print(i)

            # remove glomeruli
            final_mask = np.logical_and(final_mask, np.invert(glom_mask))
            final_mask = np.logical_and(
                final_mask, np.invert(self.segmentations['vessels']))
            
            # edit borders: 
            # remove cd31 channel from all tubules 
            all_tubs = np.logical_or(temp_mask, aqp1_mask)
            tubule_filter = np.logical_and(all_tubs, np.logical_not(self.masks['inter']))

            # if pixel is in dilated tubule mask and all tubules, add it 
            dilated_mask = binary_dilation(final_mask, disk(7)) # choose 7 as it is slightly smaller than closing on interstism
            tubule_to_add = np.logical_and(dilated_mask, tubule_filter)
            final_mask = np.logical_or(final_mask, tubule_to_add)

            print('there are ', count_draw)
            self.segmentations['tubule'] = final_mask

            print('Saving...')
            with open(os.path.join(self.cache_dir, 'tub_seg.npy'), 'wb') as f:
                np.save(f, final_mask)

        else:
            self.segmentations['tubule'] = np.load(
                os.path.join(self.cache_dir, 'tub_seg.npy'))

    def segment_nuclei(self):

        if not os.path.exists(os.path.join(self.cache_dir, 'cell_segmentation_gauss_blur.npy')):

            print('not yet implemented')
            pass
        else:
            self.segmentations['nuclei'] = np.load(os.path.join(
                self.cache_dir, 'cell_segmentation_gauss_blur.npy'))

    def validate_mask_pixels(self, mask_file, gt_paths):
        mets_labels = ["accuracy", "precision", "recall"]
        mets = []
        seg_mask = np.load(os.path.join(self.cache_dir, mask_file))

        # for each file, calculate metrics
        for abs_file in gt_paths:
            # Open the image using PIL (Pillow)
            gt_mask = Image.open(abs_file)

            # convert the mask to grayscale
            gt_mask = ImageOps.grayscale(gt_mask)

            # Convert the PIL image to a NumPy array
            gt_mask = np.array(gt_mask)

            file = re.split(r'[/]', abs_file)

            mask_coord = [float(elem)
                          for elem in re.split(r'[_]', file[-1])[1:5]]

            seg_patch = seg_mask[int(mask_coord[1]):(int(mask_coord[1]) + int(
                mask_coord[3])), int(mask_coord[0]): int(mask_coord[0]) + int(mask_coord[2])]
            seg_patch = seg_patch.astype(np.integer)

            metrics = pixelwise_metrics(gt_mask, seg_patch)

            mets.append(metrics)

        # return metrics as pandas datafrmae
        return pd.DataFrame(mets, columns=mets_labels)

    def validate_mask_element(self, mask_file, gt_paths, iou_thresh=0.5):

        mets = []
        mets_labels = ['precision', 'recall', 'F1', 'avg_matched_iou',
                       'panoptic_quality', 'num prediction objects', 'num gt objects']

        seg_mask = np.load(os.path.join(self.cache_dir, mask_file))

        # for each file, calculate metrics
        for abs_file in gt_paths:
            # Open the image using PIL (Pillow)
            gt_mask = Image.open(abs_file)

            # convert the mask to grayscale
            gt_mask = ImageOps.grayscale(gt_mask)

            # Convert the PIL image to a NumPy array
            gt_mask = np.array(gt_mask)

            file = re.split(r'[/]', abs_file)

            mask_coord = [float(elem)
                          for elem in re.split(r'[_]', file[-1])[1:5]]

            seg_patch = seg_mask[int(mask_coord[1]):(int(mask_coord[1]) + int(
                mask_coord[3])), int(mask_coord[0]): int(mask_coord[0]) + int(mask_coord[2])]

            seg_patch = seg_patch.astype(np.integer)

            metrics = matched_detection_metrics(
                gt_mask, seg_patch, iou_thresh=iou_thresh)

            mets.append(metrics)

        # return metrics as pandas datafrmae
        return pd.DataFrame(mets, columns=mets_labels)

    def validate_mask(self, mask_file, gt_paths):
        # load tubule mask
        seg_mask = np.load(os.path.join(self.cache_dir, mask_file))

        mets = []
        mets_labels = ["accuracy", "precision", "recall", "f1", "iou"]

        # for each file, calculate metrics
        for abs_file in gt_paths:
            # Open the image using PIL (Pillow)
            gt_mask = Image.open(abs_file)

            # convert the mask to grayscale
            gt_mask = ImageOps.grayscale(gt_mask)

            # Convert the PIL image to a NumPy array
            gt_mask = np.array(gt_mask)

            file = re.split(r'[/]', abs_file)

            mask_coord = [float(elem)
                          for elem in re.split(r'[_]', file[-1])[1:5]]

            seg_patch = seg_mask[int(mask_coord[1]):(int(mask_coord[1]) + int(
                mask_coord[3])), int(mask_coord[0]): int(mask_coord[0]) + int(mask_coord[2])]

            # convert seg_patch to binary if necessary
            if len(np.unique(seg_patch)) > 2:
                seg_patch[seg_patch > 0] = np.max(seg_patch)
            # convert gt mask to binary
            gt_mask[gt_mask > 0] = np.max(seg_patch)

            metrics = get_mask_metrics(gt_mask, seg_patch)

            mets.append(metrics)

        # return metrics as pandas datafrmae
        return pd.DataFrame(mets, columns=mets_labels)

    def plot_gt_on_channel(self, gt_file, channel_name, file_path):
        # load ground truth mask
        # Open the image using PIL (Pillow)
        gt_mask = Image.open(gt_file)

        # convert the mask to grayscale
        gt_mask = ImageOps.grayscale(gt_mask)

        # Convert the PIL image to a NumPy array
        gt_mask = np.array(gt_mask)

        # get corresponding region on channel
        file = re.split(r'[/]', gt_file)

        mask_coord = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

        channel = self.mif[self.channel_dict[channel_name], :, :]

        channel = channel[int(mask_coord[1]):(int(mask_coord[1]) + int(mask_coord[3])),
                          int(mask_coord[0]): int(mask_coord[0]) + int(mask_coord[2])]

        # ensure mask is integer
        channel = channel.astype('float64')
        gt_mask = gt_mask.astype('float64')

        gt_mask *= 255 / np.max(gt_mask)
        channel *= 255 / np.max(channel)

        plot_mask(gt_mask, channel, 0.7, file_path)

    def plot_mask_on_gt(self, mask_file, gt_file, file_path, outline=False, weight=0):
        # load ground truth mask
        # Open the image using PIL (Pillow)
        gt_mask = Image.open(gt_file)

        # convert the mask to grayscale
        gt_mask = ImageOps.grayscale(gt_mask)

        # Convert the PIL image to a NumPy array
        gt_mask = np.array(gt_mask)

        # get corresponding region on channel
        file = re.split(r'[/]', gt_file)

        plt.imsave('nuc_gt.png', gt_mask)

        mask_coord = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

        seg_mask = np.load(os.path.join(self.cache_dir, mask_file))

        seg_mask = seg_mask[int(mask_coord[1]):(int(mask_coord[1]) + int(mask_coord[3])),
                            int(mask_coord[0]): int(mask_coord[0]) + int(mask_coord[2])]

        # add oulines to segmentation mask

        seg_mask_outline = np.zeros_like(seg_mask, dtype=np.uint8)
        seg_mask_ext = np.copy(seg_mask)
        seg_mask_ext.dtype = 'uint8'
        plt.imsave('seg_mask.png', seg_mask)
        ext_contours, heirarchy = cv2.findContours(image=seg_mask_ext, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(seg_mask_outline, ext_contours, -1, 255, weight)
        print('nonzero outline ', np.count_nonzero(seg_mask_outline))

        # ensure mask is integer
        seg_mask = seg_mask.astype('float64')
        gt_mask = gt_mask.astype('float64')

        gt_mask *= 255 / np.max(gt_mask)
        seg_mask *= 255 / np.max(seg_mask)

        channel_rgb = np.zeros((gt_mask.shape[0], gt_mask.shape[1], 3))
        channel_rgb[:, :, 0] = gt_mask
        channel_rgb[:, :, 1] = gt_mask
        channel_rgb[:, :, 2] = gt_mask

        seg_rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3))
        seg_rgb[:, :, 0] = seg_mask
        if outline:
            print('adding outline')
            seg_outline_rgb = np.zeros(
                (seg_mask_outline.shape[0], seg_mask_outline.shape[1], 3))
            seg_outline_rgb[:, :, 2] = seg_mask_outline
        else:
            seg_outline_rgb = None

        plot_mask(channel_rgb, seg_rgb, 0.5, file_path,
                  mask_outline=seg_mask_outline)

    def plot_mask(self, mask_file, patch, file):
        seg_mask = np.load(os.path.join(self.cache_dir, mask_file))

        seg_mask = seg_mask[int(patch[1]):(int(patch[1]) + int(patch[3])),
                            int(patch[0]): int(patch[0]) + int(patch[2])]

        print('nonzero ', np.count_nonzero(seg_mask))
        print(seg_mask.shape)

        print(np.max(seg_mask))

        plt.imsave(file, seg_mask)

    def plot_mask_on_channel(self, mask_file, first_channels, first_channel_color, second_channels, second_channel_color, patch, file_path, outline=None, weight=0):

        seg_mask = np.load(os.path.join(self.cache_dir, mask_file))

        seg_mask = seg_mask[int(patch[1]):(int(patch[1]) + int(patch[3])),
                            int(patch[0]): int(patch[0]) + int(patch[2])]

        first_channel = np.zeros_like(
            seg_mask, dtype=self.masks[first_channels[0]].dtype)

        for channel_name in first_channels:

            channel = self.mif[self.channel_dict[channel_name], :, :]

            channel = channel[int(patch[1]):(int(patch[1]) + int(patch[3])),
                              int(patch[0]): int(patch[0]) + int(patch[2])]

            first_channel = np.maximum(first_channel, channel)

        second_channel = np.zeros_like(
            seg_mask, dtype=self.masks[second_channels[0]].dtype)

        for channel_name in second_channels:

            channel = self.mif[self.channel_dict[channel_name], :, :]

            channel = channel[int(patch[1]):(int(patch[1]) + int(patch[3])),
                              int(patch[0]): int(patch[0]) + int(patch[2])]

            second_channel = np.maximum(second_channel, channel)

        # add oulines to segmentation mask
        seg_mask_outline = np.zeros_like(seg_mask, dtype=np.uint8)
        seg_mask_ext = np.copy(seg_mask)
        seg_mask_ext.dtype = 'uint8'
        ext_contours, heirarchy = cv2.findContours(image=seg_mask_ext, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(seg_mask_outline, ext_contours, -1, 255, weight)

        # ensure mask is integer
        second_channel = second_channel.astype('float64')
        first_channel = first_channel.astype('float64')
        seg_mask = seg_mask.astype('float64')

        second_channel *= 255 / np.max(second_channel)
        first_channel *= 255 / np.max(first_channel)
        seg_mask *= 255 / np.max(seg_mask)

        channel_rgb = np.zeros(
            (first_channel.shape[0], first_channel.shape[1], 3))
        channel_rgb[:, :, 0] = first_channel * \
            first_channel_color[0] / np.max(first_channel)
        channel_rgb[:, :, 1] = first_channel * \
            first_channel_color[1] / np.max(first_channel)
        channel_rgb[:, :, 2] = first_channel * \
            first_channel_color[2] / np.max(first_channel)

        channel_rgb[:, :, 0] += second_channel * \
            second_channel_color[0] / np.max(second_channel)
        channel_rgb[:, :, 1] += second_channel * \
            second_channel_color[1] / np.max(second_channel)
        channel_rgb[:, :, 2] += second_channel * \
            second_channel_color[2] / np.max(second_channel)

        seg_rgb = np.zeros((seg_mask.shape[0], seg_mask.shape[1], 3))
        seg_rgb[:, :, 0] = seg_mask
        seg_rgb[:, :, 1] = seg_mask
        seg_rgb[:, :, 2] = seg_mask

        if outline:
            seg_outline_rgb = np.zeros(
                (seg_mask_outline.shape[0], seg_mask_outline.shape[1], 3))
            seg_outline_rgb[:, :, 2] = seg_mask_outline
        else:
            seg_outline_rgb = None

        plot_mask(channel_rgb, seg_rgb, 0.5, file_path,
                  mask_outline=seg_mask_outline)

    def plot_mask_on_he(self, mask_file: str, he_file: str, patch: list, output_file: str, mpp: float = 0.5):

        mask = np.load(os.path.join(
            self.cache_dir + '/registrations', mask_file))

        mask_size = np.shape(mask)

        print('mask size ', mask_size)

        mask = mask[int(patch[1]): int(patch[1]) +
                    int(patch[3]), int(patch[0]): int(patch[0]) + int(patch[2])]

        plt.imsave('test_registration.png', mask)

        # convert mask to 3 channels
        seg_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
        seg_rgb[:, :, 2] = mask

        # he = open_slide(he_file)

        # he = get_patch_at_mpp(
        #     wsi=he,
        #     mpp=mpp,
        #     coords_level0=(0, 0),
        #     patch_size_mpp=mask_size,
        # )

        # print(he.level_dimensions)
        # he = he.read_region(
        #     (0, 0), 0, (51792, 38955))
        he = np.load(os.path.join(
            self.cache_dir + '/registrations', 'he_mpp.npy'))

        he = np.array(he, dtype='uint8')

        print('before resize ', he.shape)
        # he = np.resize(he, (24401, 31318, 4))

        he_wsi = he[int(patch[1]): int(patch[1]) +
                    int(patch[3]), int(patch[0]): int(patch[0]) + int(patch[2]), :]

        # he_wsi = np.zeros((mask.shape[0], mask.shape[1], 3), dtype='uint8')
        # he_wsi[:, :, 0] = he[:, :, 0]
        # he_wsi[:, :, 1] = he[:, :, 1]
        # he_wsi[:, :, 2] = he[:, :, 2]
        # plt.imsave('test_he.png', he_wsi)
        print(he_wsi.shape)
        print(np.count_nonzero(mask))
        seg_rgb.dtype = 'uint8'
        print(seg_rgb.shape)

        seg_mask_outline = np.zeros_like(mask, dtype=np.uint8)
        seg_mask_ext = np.copy(mask)
        seg_mask_ext.dtype = 'uint8'
        ext_contours, heirarchy = cv2.findContours(image=seg_mask_ext, mode=cv2.RETR_EXTERNAL,
                                                   method=cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(seg_mask_outline, ext_contours, -1, 255, 5)

        plot_mask(he_wsi, seg_rgb, 0.5, output_file,
                  mask_outline=seg_mask_outline)

    def register_mask(self, mask_file: str, he: str, homog: str, output_file: str, mif_mpp_level0: float = 0.5077, mpp4align: float = 0.5) -> None:
        """ Function to transform a mask given a homography
        transformation

        Args:
            mask_file (str): location of mask
            he (str): location of H&E WSi
            homog (str): location of homography keypoints
            output_file (str): location to output the file 
            mif_mpp_level0: MPP for level 0 of MIF 
            mpp4aligm: the mpp we want to work with when aligning the images
        """
        mask = np.load(mask_file)

        mif_patch_size = (mask.shape[0], mask.shape[1])

        homog_kpts = load(homog)

        he_wsi = open_slide(he)

        mask.dtype = np.uint8

        mif_wsi_mask = WSI(img=mask, mpp_level0=mif_mpp_level0)

        src_patch, src_aligned_patch, targ_bbox, infon = \
            get_src_patch_aligned_to_targ_bbox(
                src_coords_L0=(0, 0),  # start at top left corner
                src_patch_size_mpp=mif_patch_size,
                mpp=mpp4align,
                src_wsi=mif_wsi_mask,
                targ_wsi=he_wsi,
                homog_kpts=homog_kpts,
                FILL_VALUE=0)

        register_dir = os.path.join(self.cache_dir, 'registrations')

        with open(os.path.join(register_dir, output_file), 'wb') as f:
            np.save(f, src_aligned_patch)

        with open(os.path.join(register_dir, 'he_mpp.npy'), 'wb') as f:
            np.save(f, targ_bbox)
