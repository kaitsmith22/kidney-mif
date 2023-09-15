YATT_PATH = 'yatt'
import sys

sys.path.append(YATT_PATH)


WSIC_PATH = 'wsic'
sys.path.append(WSIC_PATH)

from stardist.models import StarDist2D
from csbdeep.utils import normalize
import cv2 as cv
from wsic import handle_overlapping_patch_segs
from wsic import h5_utils
from wsic import cell_info
from wsic import CellSegLoader
from wsic import CellSegSaver
import tifffile
from random import sample
from skimage.exposure import rescale_intensity
from yatt.viz import plot_patch_on_wsi
from yatt.IntensityRangeRescaler import IntensityRangeRescaler, fit_rescaler_on_patches
from yatt.tissue_mask import IntensityTissueMasker
from yatt.MaskAndPatch import MaskAndPatch
from yatt.WSI import WSI
from yatt.read_image import get_wsi_at_mpp, get_patch_at_mpp
import os
import cv2

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time
from stardist.plot import render_label
from skimage.filters import threshold_li, threshold_otsu, threshold_multiotsu
from PIL import Image
import importlib



csmp = importlib.import_module("cellseg-models-pytorch")


Image.MAX_IMAGE_PIXELS = None


load_from_cache = False


def load_mif_image(fpath, level=0, ch_idx=None):

    start_time = time()
    with tifffile.TiffFile(fpath) as tif:
        pyramid = list(reversed(sorted(tif.series, key=lambda p: p.size)))
        size = pyramid[0].size  # size at level 0
        pyramid = [p for p in pyramid if size % p.size == 0]
        pyramid = [p.asarray() for p in pyramid]

    img = pyramid[level]
    if ch_idx is not None:
        img = img[ch_idx, :, :]

    print("Opening {} took {:1.3f} seconds".format(fpath, time()-start_time))

    return img


data_dir = 'data'
fname = '213_HIVE3_TMA_191_7_10_6_22_Scan2.qptiff'
mif_fpath = os.path.join(data_dir, fname)
mif_dapi_fpath = os.path.join(
    data_dir, '213_HIVE3_TMA_191_7_10_6_22_Scan2-dapi.png')

# Note MIF level 0 is at 20x
MIF_MPP_LEVEL_0 = 0.5077

# we cache some of these opeations because they take a while
if load_from_cache:
    mif_dapi_img = np.array(Image.open(mif_dapi_fpath))
else:
    mif_dapi_img = load_mif_image(mif_fpath, ch_idx=0)
    Image.fromarray(mif_dapi_img).save(mif_dapi_fpath)

# # get cd45
# mif_cd45_orig = load_mif_image(mif_fpath, ch_idx=6)
#
# # threshold channel
# cd45 = 255 * cv2.normalize(mif_cd45_orig, None, 1, cv2.NORM_L2, dtype=cv2.CV_32F)
# cd45_mask = np.full_like(cd45, False, dtype=bool)
# otsu_thresh = threshold_otsu(cd45)
# cd45_mask[cd45 >= otsu_thresh] = True
#
# print('num nonzero ', np.count_nonzero(cd45_mask))
#
# mif_dapi_img[cd45_mask != True] = 0
# print('num nonzero ', np.count_nonzero(mif_dapi_img))
# print('loaded img')

# wrap the numpy array in a WSI object
wsi = WSI(img=mif_dapi_img, mpp_level0=MIF_MPP_LEVEL_0)

# visualize the WSI at low power
img4viz = get_wsi_at_mpp(wsi, mpp=10)

patch_size = (256, 256)  # patch size to read in
min_tissue_prop = 0.2  # min proportion tissue to include patch

overlap = 50

# create patch grid
patcher = MaskAndPatch(mpp4mask=10,
                       tissue_masker=IntensityTissueMasker(),
                       mpp4patch=MIF_MPP_LEVEL_0,
                       patch_size=(256, 256)
                       )

patcher.fit_mask_and_patch_grid(wsi=wsi)

# find the patches with enough tissue
patch_idxs_with_tissue = patcher.get_patch_idxs_with_tissue(min_tissue_prop)
print("{} patches have tissue".format((len(patch_idxs_with_tissue))))

# examine patch grid + tissue mask
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.imshow(~patcher.mask_, cmap='gray')
plt.title("tissue mask")
plt.axis('off')

plt.subplot(1, 2, 2)
patcher.draw_wsi_with_patch_grid(
    min_tissue_prop=min_tissue_prop, thickness=1, show_ticks=True)
plt.title("Patches with at least {}% tisssue".format(100*min_tissue_prop))

# rescale the images to enhance contrast
rescaler = IntensityRangeRescaler(min_q=1, max_q=99, to01=True, max_range=255)

# this samples a few pathes at high power to compute the 99th percentile of the intensities
rescaler, _, runtimes = \
    fit_rescaler_on_patches(wsi=wsi,
                            rescaler=rescaler,
                            masker_patcher=MaskAndPatch(mpp4patch=MIF_MPP_LEVEL_0,
                                                        tissue_masker=IntensityTissueMasker()),
                            n_per_bin=10,
                            random_state=2)

print("Estimated largest percentile", rescaler.max_val_)

in_range = (rescaler.min_val_, rescaler.max_val_)
out_range = (0, rescaler.max_range)

mif_dapi_img = rescale_intensity(image=mif_dapi_img,
                                 in_range=in_range,
                                 out_range=out_range)


# get multi otsu thresholding
thresh = threshold_multiotsu(mif_dapi_img, classes=3)
mif_dapi_img[mif_dapi_img < thresh[1]] = 0
mif_dapi_img = mif_dapi_img.astype(float)

wsi_thresh = WSI(img=mif_dapi_img.astype('uint8'), mpp_level0=MIF_MPP_LEVEL_0)

print('threshold was ', thresh)

# seg_fpath = 'no_overlap.h5'
# saver = CellSegSaver.CellSegSaver(fpath=seg_fpath, delete_if_exists=True)
#
# overlapping_patch_size = patch_size
seg_fpath = 'overlapping.h5'
saver = CellSegSaver.CellSegSaver(fpath=seg_fpath, delete_if_exists=True)

overlapping_patch_size = (patch_size[0] + overlap, patch_size[1] + overlap)

plot = False


def get_segmentations(mif_dapi_img, blur, resize, prob):
    seg_fpath = 'overlapping.h5'
    saver = CellSegSaver.CellSegSaver(fpath=seg_fpath, delete_if_exists=True)

    overlapping_patch_size = (patch_size[0] + overlap, patch_size[1] + overlap)
    # model = StarDist2D.from_pretrained('2D_versatile_fluo')
    # model.thresholds = {'prob': prob, 'nms': 0.3}
    model = csmp.models.cellpose_base(type_classes=1)

    index_count = 0
    for i in tqdm(range(len(patch_idxs_with_tissue))):
        patch_idx = patch_idxs_with_tissue[i]
        # patch_idx = i
        loc_level0 = patcher.patch_coords_level0_[patch_idx]

        runtimes = {}
        ##########################
        # Load and process patch #
        ##########################
        start_time = time()

        patch_orig = get_patch_at_mpp(wsi=wsi,
                                      mpp=MIF_MPP_LEVEL_0,
                                      coords_level0=loc_level0,
                                      patch_size_mpp=overlapping_patch_size)

        patch = get_patch_at_mpp(wsi=wsi_thresh,
                                 mpp=MIF_MPP_LEVEL_0,
                                 coords_level0=loc_level0,
                                 patch_size_mpp=overlapping_patch_size)

        patch = cv2.GaussianBlur(patch, (blur, blur), 0)
        patch_rescaled = patch / rescaler.max_range
        # patch_rescaled = rescaler.rescale(patch)
        # thresh = threshold_otsu(patch_rescaled)
        # patch_rescaled[patch_rescaled < thresh] = 0

        runtimes['runtime_preprocess'] = time() - start_time

        ####################
        # Run segmentation #
        ####################

        start_time = time()

        # label_img = multi_otsu_seg(patch_rs, classes=3, sigma=.4)
        # label_img, details = model.predict_instances(patch_rescaled.astype(np.float32), prob_thresh = prob, scale = resize)
        label_img = model(patch_rescaled.astype(np.float32))
        label_img = label_img['cellpose']

        # add 1 to avoid "collision" with largest label for patch and smallest label for next patch
        label_img = label_img + index_count
        index_count += len(np.unique(label_img))

        runtimes['runtime_segment'] = time() - start_time

        if plot:
            plt.subplot(1, 2, 2)
            file = 'seg_test/patch' + str(i) + '.png'
            plt.imsave(file, render_label(label_img, img=patch_orig))

        ################
        # Save results #
        ################

        patch_info = {'patch_loc_x': loc_level0[0],
                      'patch_loc_y': loc_level0[1],
                      'patch_grid_row': int(loc_level0[1] // patch_size[1]),
                      'patch_grid_col': int(loc_level0[0] // patch_size[0]),
                      **runtimes}

        saver.save_labeled_patch(
            labeled_patch=label_img, patch_info=patch_info)

    seg_fpath = 'merged_overlapping.h5'

    merged_loader = CellSegLoader.CellSegLoader('overlapping.h5')

    merged_saver = CellSegSaver.CellSegSaver(
        fpath=seg_fpath, delete_if_exists=True)

    handle_overlapping_patch_segs.save_merged_segmentations(
        merged_saver, merged_loader, merged_loader.load_patch_info(), overlap)

    merged_loader = CellSegLoader.CellSegLoader('merged_overlapping.h5')

    merged_np = np.array(merged_loader.load_region_seg(
        (0, 0), (24000, 30960), 256))

    np.save('cell_segmentation' + str(blur) + '_' +
            str(resize) + '_' + str(prob) + '.npy', merged_np)


blurs = [3, 7, 9]
probs = [0.3, 0.5]
resizes = [0.59, 1, 2]


for blur in blurs:
    for prob in probs:
        for resize in resizes:
            if blur != 3 and probs != 0.3 and resizes != 0.59:
                get_segmentations(mif_dapi_img, blur=blur,
                                  resize=resize, prob=prob)
# plt.imsave('cell_v1.png', merged_loader.load_region_seg((11264, 2304), (1000, 1000), 256))
