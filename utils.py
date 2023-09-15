"""
Helper function for MIF_Segmentation object
"""
import sys
from time import time
import tifffile
import numpy as np

import numpy as np
YATT_PATH = r'yatt'
sys.path.append(YATT_PATH)
WSI_PATH = r'wsi_reg'
sys.path.append(WSI_PATH)

import cv2 as cv
import matplotlib.pyplot as plt
from wsi_reg.utils import get_intensity_mask, rescale_intensity
from skimage.filters import threshold_li, threshold_otsu, threshold_multiotsu
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score

def plot_mask(gt, mask, weight, file_path, mask_outline = None, color = (0,0,127)):

    # # convert numpy arrays to cv2
    # mask = cv.fromarray(mask)
    # gt = cv.fromarray(mask

    # add weighted
    img = cv.addWeighted(gt, 1, mask, weight, 0)

    # add outline
    if type(mask_outline) == np.ndarray:
        img = np.array(img)

        img[mask_outline == 255] = color

    cv.imwrite(file_path, img)



def get_mask_metrics(gt_mask, seg_mask):

    # Flatten the masks to 1D arrays (needed for some metrics)
    ground_truth_flat = gt_mask.flatten()
    seg_flat = seg_mask.flatten()

    # Calculate Accuracy
    accuracy = accuracy_score(ground_truth_flat, seg_flat)

    # Calculate Precision
    precision = precision_score(ground_truth_flat, seg_flat, pos_label = np.max(ground_truth_flat))

    # Calculate Recall
    recall = recall_score(ground_truth_flat, seg_flat, pos_label = np.max(ground_truth_flat))

    # Calculate F1 Score
    f1 = f1_score(ground_truth_flat, seg_flat, pos_label = np.max(ground_truth_flat))

    # Calculate Intersection over Union (Jaccard Score)
    iou = jaccard_score(ground_truth_flat, seg_flat, pos_label = np.max(ground_truth_flat))
    return(accuracy, precision, recall, f1, iou)

def load_mif(fpath):
    """
    Parameters
    -----------
    fpath: str
        Path to the image.

    xy: tuple of ints
        The xy coordinates of the patch in the full image.

    patch_size: tuple of ints
        The size of the patch in the full image. Note this should be yx.
    """


    # load the whole image
    start_time = time()
    with tifffile.TiffFile(fpath) as tif:
        pyramid = list(reversed(sorted(tif.series, key=lambda p:p.size)))
        size = pyramid[0].size
        pyramid = [p for p in pyramid if size % p.size == 0]
        pyramid = [p.asarray() for p in pyramid]

    level = 0
    img = pyramid[level]
    del pyramid
    print("Loading image took", time() - start_time)

    return img


def get_mask(mif, lower_thresh, upper_thresh, method='otsu'):

    if method == 'adaptive':
        inter_n = rescale_intensity(mif, p=(lower_thresh, upper_thresh))
        mask = cv.adaptiveThreshold(inter_n, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                                     cv.THRESH_BINARY, 199, -3)

    elif method == 'otsu':
        inter_n = 255 * cv.normalize(mif, None, 1, cv.NORM_L2, dtype=cv.CV_32F)
        mask = np.full_like(inter_n, False, dtype=bool)
        otsu_thresh = threshold_otsu(inter_n)
        mask[inter_n >= otsu_thresh] = True

    elif method == 'multi-otsu':
        inter_n = 255 * cv.normalize(mif, None, 1, cv.NORM_L2, dtype=cv.CV_32F)
        mask = np.full_like(inter_n, False, dtype=bool)
        otsu_thresh = threshold_multiotsu(inter_n)
        mask[inter_n >= otsu_thresh[0]] = True

    else:
        inter_n = 255 * cv.normalize(mif, None, 1, cv.NORM_L2, dtype=cv.CV_32F)
        mask = np.full_like(inter_n, False, dtype=bool)

        pixl_thresh_upper = np.percentile(inter_n.ravel(), q=upper_thresh)
        pixl_thresh_lower = np.percentile(inter_n.ravel(), q=lower_thresh)

        mask[(inter_n >= pixl_thresh_lower) & (inter_n < pixl_thresh_upper)] = True
    return mask
