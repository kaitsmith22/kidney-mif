# kidney-mif

Repository for project interested in computing quantitative metrics for normal and abnormal kidney cell structures using mIF and H&E data.

## Project Structure
- MIF_Segmentation: class to handle segmentation of a MIF WSI
- run_pipeline_final.py: script to handle pipeline to create segmentation masks 
- utils.py: general util functions used in MIF segmentation pipeline, like thresholding the MIF channels
- evaluate_masks.py: Get metrics for various masks
- plot_masks.py: Generate plots to qualitatively evaluate masks
- compute_tubule_int_ratio: initial script to compute the tubule interstism ratio
- compute_capillary_density: script to compute the density for each core biopsy
- gt_masks_final: directory with annotations I made for nuclei, tubules, glomeruli, and interstism. My nuclei masks should really be evaluated/replaced by a pathologist's annotations
- save_annoations.groovy: script to save annoations in Qupath
 

NOTE: I currently have my directory structured as follows: 
- data: directory with MIF and H&E files
- masks: directory where the masks are saved (note that masks are saved in a sub-folder with the same name as the file you are creating the masks on - in this case: `/213_HIVE3_TMA_191_7_10_6_22_Scan2`).
---- registrations: directory with registered masks

## Using the Registered Masks in H&E Segmentation Algorithm Training

In order to use the registered masks with the H&E files, you need to open the H&E files at the correct MPP. Use the YATT package to do so:

```
    he = open_slide(he_file)
     he = get_patch_at_mpp(
         wsi=he,
         mpp=0.5,
         coords_level0=(0, 0),
         patch_size_mpp=mask_size,
     )
```

## Current Metrics for Mask Evaluation 

Metrics are calculated in 2 ways. First, we compute precision, accuracy, and recall as you would expect,
on a pixel-by-pixel basis. In addition, for glomeruli, tubules, and nuclei, we first match elements in the masks with elements in the annotations by matching elements with an IOU score of at least 0.5. Then we compute precion, recall and F1 on these matched elements.

Below are the current metrics for the masks: 

```
----------Glomeruli Metrics Pixel----------
   accuracy  precision    recall
0  0.916993   0.936065  0.978265
1  0.733080   0.916966  0.785204
2  0.873902   0.955599  0.910889
3  0.739483   0.957208  0.764764
----------Glom Metrics Element Wise ----------
   precision    recall        F1  avg_matched_iou  panoptic_quality  num prediction objects  num gt objects
0       1.00  1.000000  1.000000         0.902726          0.902726                       3               3
1       0.75  1.000000  0.857143         0.746740          0.640063                       4               3
2       1.00  1.000000  1.000000         0.851917          0.851917                       3               3
3       1.00  0.833333  0.909091         0.923930          0.839937                       5               6
----------Interstism Metrics----------
   accuracy  precision    recall
0  0.605967   0.771656  0.738367
1  0.738806   0.891598  0.811719
2  0.805250   0.904998  0.879604
----------Tubule Metrics Pixel----------
   accuracy  precision    recall
0  0.868115   0.917475  0.941643
1  0.757079   0.893586  0.832099
----------Tub Metrics Element Wise ----------
   precision    recall        F1  avg_matched_iou  panoptic_quality  num prediction objects  num gt objects
0   0.590909  0.709091  0.644628         0.770667          0.496794                      66              55
1   0.741935  0.836364  0.786325         0.806937          0.634514                      62              55
```

## Current Metrics 

### Tubule - Interstitium Ratio

For each core biopsy, compute the ratio of the number of tubule pixels, to the number of pixels in the interstitium mask.

### Compute Capillary Density  

For each core biopsy, compute the ratio of the number of pixels in the capillary mask to the number of pixels in the core biopsy. 

## MIF Segmentation Pipeline

The segmentation pipeline creates the following masks (at the moment, more to come):
1. Core biopsy mask (core_biopsy.npy)
    This mask segments out the core biopsies (or as we scientifically call them, tissue strips) for each WSI. Currently, this number needs to be set manually
2. Interstitium mask (inter_seg.npy)
    This is the mask for the interstitium
3. Glomeruli mask (initial_glom.npy)
   Mask for the glomeruli
4. Vessel mask (vessels.npy)
    Mask for the objects in the SMA (smooth muscle) channel 
5. Tubule mask (tub_seg.npy)
    This is the mask for ALL of the tubules
6. Capillary mask (capillary_seg.npy)
   

    
### Notes on MIF Channel Thresholding

For the majority of the channels, otsu thresholding performed well enough. However, for the interstism, the channel first needs to have the intensity rescaled to only include the pixels from the 95th to 99th percentile range. If the intensity is not rescaled, too many pixels are labeled as interstism, and a very noisy mask is created. This is particularly problematic around the edges of the WSI. Obviously, none of the pixels outside of the core biopsies should be labeled as positive for any of the MIF channels. Hence, rescaling the pixel distribution removes the unnecessary signal. 

Next adapative thresholding is applied. The signal on the interstism does not express equal intensity across all core biopsies, nor is it evenly expressed within a single core biopsy. To account for this, CV2 implementation of adaptive thresholding adjusts the threshold value for local regions. After experimentation, a block size of 199 was chosen for the hyperparameter for adaptive thresholding.
    
### Computing the Core Biopsy Mask

After thresholding the individual channels in the MIF, the next step in the segmention pipeline is to segment the core biopsies. This is done with the following steps:
1. Add all MIF channels used in the pipeline to a temporary mask
2. Close any holes with an area smaller than $500^2$ pixels 
3. Open any holes with an area smaller than 5000 pixels

### Computing the Vessel Mask
1. Find all contours in the SMA channel
2. Fill in regions (i.e. we want to fill in the holes in the center of the vessels)

### Computing the Glomeruli Mask
1. Remove vessels from CD34 channel
2. Perform binary closing (with disk of size 3 pixels)
3. Remove all regions with an area smaller than 1000 pixels (this removes noise, as gloms are large structures)
4. Find all holes in the combination of the tubule and interstism masks (i.e. we aren't interested in the holes in the interstitium mask that are tubules, so don't look at those)
5. If the hole on the interstitium mask has any CD34 pixels, check if the region is circular
6. If it is, add it to the mask

### Computing the Interstism Mask 

Next, the interstism mask is created, by following these steps:
1. Remove the CD31 and CD34 (these are signals that are expressed in the glomeruli, which we don't want in the interstism mask) from the MIF collagen channel
2. Remove vessels 
3. Open (ie remove) any region smaller than $50^2$ pixels
4. Perform closing with a disk size of 6. This helps close any gaps in the interstism
5. Close any holes smaller than 500 pixels in area. This step includes the insterstisial capillaries in the interstism mask. I'm not sure if this is correct

### Computing the Tubule Mask

Finally the tubule mask is created by:
1. Combining all thresholded tubule channels together. Specifically, this is AQP1, AQP2, CK7, Calbindin, and Panck.
2. Removing the glomeruli and vessels from the tubule channels 
3. Drawing the outline of the core biopsy mask on the interstism mask. This allows for us to segment out the tubules on the edge of the core biopsies
4. For each hole in the interstism mask, check if a enough tubule markers exist in this region. If so, draw this region on the tubule mask. Currently, 10% of the region needs to have AQP2, CK7, Calbindin, or PanCK, or 20% of the region needs to have AQP1. You'll find that when a tubule expresses AQP1, a lot more signal is present in the region, where tubules that express the other markers do so at much lesser intensities. 


### Computing the Capillary Mask 
1. Remove the glomeruli and vessels from the CD31 channel. 

### Notes on MIF_segmentation.py

This class handles all of the mask generation, validation, registraion, and plotting. I'll be honest, this file has grown a little out of hand. Feel free to re-structure as you see fit. 


## Next Steps 
- Validate/improve nuclear segmentation
- Count number of immune cells
      - Get nuclei with CD31 or CD34 present in the cells
- Validate and potentially perfect the capillary mask. Use pathologist annotations for capillaries. 



