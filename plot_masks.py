"""
Script to run segmentation pipeline
"""
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from MIF_Segmentation import MIF_Segmentation

channel_dict = {
    "inter": 13,
    "aqp1": 15,
    "aqp2": 16,
    "ck7": 3,
    "cal": 9,
    "panck": 8,
    "cd31": 5,
    "cd34": 4,
    "sma": 12,
    "dapi": 1
}

threshold_dict = {
    "inter": (95, 99),
    "aqp1": (93, 99),
    "aqp2": (99, 100),
    "ck7": (99.9, 100),
    "cal": (99, 100),
    "panck": (99, 100),
    "cd31": (1, 99),
    "cd34": (99, 100),
    "sma": (1, 99),
    "dapi": (1, 99)
}

method_dict = {
    "inter": 'adaptive',
    "aqp1": 'otsu',
    "aqp2": 'otsu',
    "ck7": 'otsu',
    "cal": 'otsu',
    "panck": 'otsu',
    "cd31": 'adaptive',
    "cd34": 'normal',
    "sma": "otsu",
    "dapi": "otsu"
}

seg = MIF_Segmentation(r'data/213_HIVE3_TMA_191_7_10_6_22_Scan2.qptiff',
                       6,
                       channel_dict,
                       threshold_dict,
                       method_dict,
                       cache_dir=r'masks',
                       save=False)

# initialize segmentation
seg.initialize()


nuclear_files = ['nuclei_6310.0_8578.0_235.0_215.0_mask.png',
                 'nuclei_9412.0_15956.0_242.0_184.0_mask.png',
                 'nuclei_12138.0_12153.0_322.0_240.0_mask.png']
nuclear_gt = [os.path.join('gt_masks_final', file) for file in nuclear_files]


for i, nuc_file in enumerate(nuclear_gt):
    file = re.split(r'[/]', nuc_file)
    patch = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

    seg.plot_gt_on_channel(
        nuclear_gt[0], 'dapi', 'final_mask_test/nuclear_gt' + str(patch[0])+'_on_dapi.png')

    seg.plot_mask_on_gt('cell_segmentation_gauss_blur.npy',
                        nuclear_gt[0], 'final_mask_test/nuclear_mask'+str(patch[0]) + '_on_gt.png', outline=False)

    seg.plot_mask_on_channel('cell_segmentation_gauss_blur.npy', ['dapi'], (255, 0, 0), [
                             "dapi"], (255, 0, 0), patch, 'final_mask_test/nuclear_mask' + str(patch[0]) + '_on_dapi.png')


glom_files = ['glom_5568.0_13755.0_1831.0_1842.0_mask.png', 'glom_9038.0_18468.0_1320.0_1582.0_mask.png',
              'glom_11044.0_8500.0_1209.0_1342.0_mask.png', 'glom_11078.0_21329.0_1395.0_1867.0_mask.png']

glom_gt = [os.path.join('gt_masks_final', file) for file in glom_files]

for i, glom_file in enumerate(glom_gt):
    file = re.split(r'[/]', glom_file)
    patch = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

    seg.plot_gt_on_channel(
        glom_gt[i], 'cd34', 'final_mask_test/glom_gt' + str(patch[0])+'_on_cd34.png')

    seg.plot_mask_on_gt('initial_glom.npy', glom_gt[i], 'final_mask_test/glom_mask'+str(
        patch[0]) + '_on_gt.png', outline=True, weight=10)

    seg.plot_mask_on_channel('initial_glom.npy', ['cd34'],  (203, 192, 255), [
                             'inter'], (200, 213, 48), patch, 'final_mask_test/glom_mask_' + str(patch[0]) + '_on_cd31.png', outline=True, weight=10)

patch = [9038, 18468, 900, 1500]
seg.plot_mask_on_he('glomeruli_registation.npy', 'data/213_HIVE3_TMA_191.7 H&E_2.svs',
                    patch, 'final_mask_test/glom_mask_on_he.png')

exit(1)

inter_files = ['interstism_16183.0_11285.0_1384.0_692.0_mask.png', 'interstism_6382.0_8869.0_651.0_894.0_mask.png',
               'interstism_8952.0_24419.0_1096.0_795.0_mask.png']
inter_gt = [os.path.join('gt_masks_final', file) for file in inter_files]

for i, inter_file in enumerate(inter_gt):
    file = re.split(r'[/]', inter_file)
    patch = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

    seg.plot_gt_on_channel(
        inter_gt[i], 'inter', 'final_mask_test/inter_on_col.png')

    seg.plot_mask_on_gt(
        'inter_seg.npy', inter_gt[i], 'final_mask_test/inter_mask_' + str(patch[0])+'_on_gt.png', outline=False)

    seg.plot_mask_on_channel('inter_seg.npy', ['inter'], (200, 213, 48), [
                             'sma'], (255, 255, 0), patch, 'final_mask_test/inter_mask_' + str(patch[0]) + '_on_col.png', outline=False)

    # also plot glomeruli mask on same patches:
    seg.plot_mask_on_channel('initial_glom.npy', ['cd34'],  (203, 192, 255), [
                             'inter'], (200, 213, 48), patch, 'final_mask_test/glom_mask_' + str(patch[0]) + '_on_cd31.png', outline=True, weight=10)


tub_files = ['tubule_8707.0_11576.0_1295.0_653.0_mask.png',
             'tubule_11281.0_16426.0_1026.0_770.0_mask.png']
tub_gt = [os.path.join('gt_masks_final', file) for file in tub_files]


for i, tub_file in enumerate(tub_gt):
    file = re.split(r'[/]', tub_file)
    patch = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

    seg.plot_mask_on_channel('tub_seg.npy', ['aqp1', 'aqp2', 'ck7', 'panck'], (0, 0, 255), [
                             "inter"], (200, 213, 48), patch, 'final_mask_test/tub_mask' + str(patch[0]) + '_on_tubs.png', outline=True, weight=5)

    seg.plot_gt_on_channel(
        tub_gt[i], 'aqp1', 'final_mask_test/tub_' + str(patch[0]) + 'on_aqp1.png')

    seg.plot_mask_on_gt('tub_seg.npy', tub_gt[i], 'final_mask_test/tub_mask_' + str(
        patch[0]) + 'on_gt.png', outline=True, weight=10)

seg.plot_mask_on_he('tubule_registation.npy', 'data/213_HIVE3_TMA_191.7 H&E_2.svs',
                    patch, 'final_mask_test/tub_mask_on_he.png')


cap_files = ['capillary_9142.0_21794.0_421.0_296.0_mask.png',
             'capillary_15212.0_5518.0_905.0_574.0_mask.png']
cap_gt = [os.path.join('gt_masks_final', file) for file in cap_files]

for i, cap_file in enumerate(cap_gt):
    file = re.split(r'[/]', cap_file)
    patch = [float(elem) for elem in re.split(r'[_]', file[-1])[1:5]]

    seg.plot_mask_on_channel('capillary_seg.npy', ['cd31'], (0, 165, 255), [
                             "inter"], (200, 213, 48), patch, 'final_mask_test/cap_mask' + str(patch[0]) + '_on_cd31.png', outline=True, weight=5)

    seg.plot_gt_on_channel(
        cap_file, 'cd31', 'final_mask_test/cap_' + str(patch[0]) + 'on_cd31.png')

    seg.plot_mask_on_gt('capillary_seg.npy', cap_file, 'final_mask_test/cap_mask_' +
                        str(patch[0]) + 'on_gt.png', outline=True, weight=5)
