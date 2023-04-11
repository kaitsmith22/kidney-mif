"""
Helper function for MIF_Segmentation object
"""
import sys
from time import time
import tifffile
import numpy as np
import cv2
YATT_PATH = r'yatt'
sys.path.append(YATT_PATH)
WSI_PATH = r'wsi_reg'
sys.path.append(WSI_PATH)

from wsi_reg.utils import get_intensity_mask, rescale_intensity
from skimage.filters import threshold_li, threshold_otsu

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
        inter_n = rescale_intensity(mif, p=(95, 99))
        mask = cv2.adaptiveThreshold(inter_n, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                     cv2.THRESH_BINARY, 199, -3)

    elif method == 'otsu':
        inter_n = 255 * cv2.normalize(mif, None, 1, cv2.NORM_L2, dtype=cv2.CV_32F)
        mask = np.full_like(inter_n, False, dtype=bool)
        otsu_thresh = threshold_otsu(inter_n)
        mask[inter_n >= otsu_thresh] = True

    else:
        inter_n = 255 * cv2.normalize(mif, None, 1, cv2.NORM_L2, dtype=cv2.CV_32F)
        mask = np.full_like(inter_n, False, dtype=bool)

        pixl_thresh_upper = np.percentile(inter_n.ravel(), q=upper_thresh)
        pixl_thresh_lower = np.percentile(inter_n.ravel(), q=lower_thresh)

        mask[(inter_n >= pixl_thresh_lower) & (inter_n < pixl_thresh_upper)] = True
    return mask