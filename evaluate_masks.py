"""
Script to run segmentation pipeline
"""
import os
from MIF_Segmentation import MIF_Segmentation

channel_dict = {
    "inter": 13,
    "aqp1": 15,
    "aqp2": 16,
    "ck7": 3,
    "cal": 9,
    "panck": 8,
    "cd31": 5,
    "cd34": 4
}

threshold_dict = {
    "inter": (95, 99),
    "aqp1": (93, 99),
    "aqp2": (99,100),
    "ck7": (99.9,100),
    "cal": (99,100),
    "panck": (99,100),
    "cd31": (1, 99),
    "cd34": (99,100)
}

method_dict = {
    "inter": 'adaptive',
    "aqp1": 'otsu',
    "aqp2": 'otsu',
    "ck7": 'otsu',
    "cal": 'otsu',
    "panck": 'otsu',
    "cd31": 'adaptive',
    "cd34": 'normal'
}

seg = MIF_Segmentation(r'data/213_HIVE3_TMA_191_7_10_6_22_Scan2.qptiff',
                       6,
                       channel_dict,
                       threshold_dict,
                       method_dict,
                       cache_dir = r'masks',
                       save = False)


### Nuclear Segmenation Evaluation ###

nuclear_files = ['nuclei_6310.0_8578.0_235.0_215.0_mask.png', 
                 'nuclei_9412.0_15956.0_242.0_184.0_mask.png',
                 'nuclei_12138.0_12153.0_322.0_240.0_mask.png']
nuclear_gt = [os.path.join('gt_masks_final', file) for file in nuclear_files]
# nuclear_metrics  = seg.validate_mask_pixels('cell_segmentation_gauss_blur.npy', nuclear_gt)
# print('-'*10 + 'Nuclear Metrics Pixel Wise ' + '-'*10)
# print(nuclear_metrics)

# nuclear_metrics  = seg.validate_mask_pixels('cell_segmentation7_0.59_0.5.npy', nuclear_gt)

# print('-'*10 + 'Nuclear Metrics Pixel Wise ' + '-'*10)
# print(nuclear_metrics)
# nuclear_metrics  = seg.validate_mask_element('cell_segmentation7_0.59_0.5.npy', nuclear_gt, iou_thresh = 0.001)

# print('-'*10 + 'Nuclear Metrics Element Wise ' + '-'*10)
# print(nuclear_metrics)

### Glomeruli Segmentation Metrics ###

glom_files = ['glom_5568.0_13755.0_1831.0_1842.0_mask.png', 'glom_9038.0_18468.0_1320.0_1582.0_mask.png',
              'glom_11044.0_8500.0_1209.0_1342.0_mask.png', 'glom_11078.0_21329.0_1395.0_1867.0_mask.png']
glom_gt = [os.path.join('gt_masks_final', file) for file in glom_files]


glom_metrics  = seg.validate_mask_pixels('initial_glom.npy', glom_gt)

print('-'*10 + 'Glomeruli Metrics Pixel' + '-'*10)
print(glom_metrics)

glom_metrics  = seg.validate_mask_element('initial_glom.npy', glom_gt, iou_thresh = 0.001)

print('-'*10 + 'Glom Metrics Element Wise ' + '-'*10)
print(glom_metrics)

### Interstitism Segmentation Metrics ###

inter_files = ['interstism_16183.0_11285.0_1384.0_692.0_mask.png', 'interstism_6382.0_8869.0_651.0_894.0_mask.png', 'interstism_8952.0_24419.0_1096.0_795.0_mask.png']
inter_gt = [os.path.join('gt_masks_final', file) for file in inter_files]

inter_metrics = seg.validate_mask_pixels('inter_seg.npy', inter_gt)

print('-'*10 + 'Interstism Metrics' + '-'*10)
print(inter_metrics)

### Tubule Segmentation Metrics ###

tub_files = ['tubule_8707.0_11576.0_1295.0_653.0_mask.png', 'tubule_11281.0_16426.0_1026.0_770.0_mask.png']
tub_gt = [os.path.join('gt_masks_final', file) for file in tub_files]

tub_metrics = seg.validate_mask_pixels('tub_seg.npy', tub_gt)
print('-'*10 + 'Tubule Metrics Pixel'+ '-'*10)
print(tub_metrics)

tub_metrics  = seg.validate_mask_element('tub_seg.npy', tub_gt, iou_thresh = 0.001)

print('-'*10 + 'Tub Metrics Element Wise ' + '-'*10)
print(tub_metrics)

### Interstisial Capillary Metrics ### 

cap_files = ['capillary_9142.0_21794.0_421.0_296.0_mask.png', 'capillary_15212.0_5518.0_905.0_574.0_mask.png']
cap_gt = [os.path.join('gt_masks_final', file) for file in cap_files]

cap_metrics = seg.validate_mask_pixels('capillary_seg.npy', cap_gt)
print('-'*10 + 'Capillary Metrics Pixel'+ '-'*10)
print(cap_metrics)

cap_metrics  = seg.validate_mask_element('capillary_seg.npy', cap_gt, iou_thresh = 0.001)

print('-'*10 + 'Capillary Metrics Element Wise ' + '-'*10)
print(cap_metrics)
