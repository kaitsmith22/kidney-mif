"""
Script to run segmentation pipeline
"""
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
    "inter": (90, 99),
    "aqp1": (93, 99),
    "aqp2": (99,100),
    "ck7": (99.9,100),
    "cal": (99,100),
    "panck": (99,100),
    "cd31": (90, 100),
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

seg = MIF_Segmentation(r'data/220e_TMA_223_10_225_10_1_31_23_Scan1.qptiff',
                       6,
                       channel_dict,
                       threshold_dict,
                       method_dict,
                       cache_dir = 'masks',
                       save = False)

# initialize segmentation
seg.initialize()

# get interstism segmentation
seg.segment_interstism()

# get tubule segementation
seg.segment_tubules()