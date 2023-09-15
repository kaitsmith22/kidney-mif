"""
Script to run segmentation pipeline
"""
from MIF_Segmentation import MIF_Segmentation
import matplotlib.pyplot as plt

channel_dict = {
    "inter": 13,
    "aqp1": 15,
    "aqp2": 16,
    "ck7": 3,
    "cal": 9,
    "panck": 8,
    "cd31": 5,
    "cd34": 4,
    "sma": 12
}

threshold_dict = {
    "inter": (95, 99),
    "aqp1": (93, 99),
    "aqp2": (99,100),
    "ck7": (99.9,100),
    "cal": (99,100),
    "panck": (99,100),
    "cd31": (1, 99),
    "cd34": (99,100),
    "sma": (1, 99)
}

method_dict = {
    "inter": 'adaptive',
    "aqp1": 'otsu',
    "aqp2": 'multi-otsu',
    "ck7": 'multi-otsu',
    "cal": 'multi-otsu',
    "panck": 'multi-otsu',
    "cd31": 'otsu',
    "cd34": 'otsu',
    "sma": "otsu"
}

patch = [12000,15000, 3000,6000]

seg = MIF_Segmentation(r'data/213_HIVE3_TMA_191_7_10_6_22_Scan2.qptiff',
                       6,
                       channel_dict,
                       threshold_dict,
                       method_dict,
                       cache_dir = 'masks',
                       frozen = False,
                       save = False)

# initialize segmentation
seg.initialize()

seg.initialize_glomeruli(channel='cd34')

seg.segment_interstism()

seg.segment_tubules()

#seg.segment_capillaries()

