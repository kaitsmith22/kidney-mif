# kidney-mif

Repository for project interested in computing quantitative metrics for normal and abnormal kidney cell structures using mIF and H&E data.

## Project Structure
- MIF_Segmentation: class to handle segmentation of a MIF WSI
- run_pipeline.py: work-in-progress script to handle the segmentation pipeline
- utils.py: general util functions used in MIF segmentation pipeline, like thresholding the MIF channels 
- compute_tubule_int_ratio: initial script to compute the tubule interstism ratio

## MIF Segmentation Pipeline

The segmentation pipeline creates the following masks (at the moment, more to come):
1. Core biopsy mask (core_biopsy.npy)
    This mask segments out the core biopsies (or as we scientifically call them, tissue strips) for each WSI. Currently, this number needs to be set manually
2. Interstism mask (int_mask.npy)
    This is the mask for the interstism
3. Tubule mask (tub_mask.npy)
    This is the mask for ALL of the tubules
    
### Notes on MIF Channel Thresholding

For the majority of the channels, otsu thresholding performed well enough. However, for the interstism, the channel first needs to have the intensity rescaled to only include the pixels from the 95th to 99th percentile range. If the intensity is not rescaled, too many pixels are labeled as interstism, and a very noisy mask is created. This is particularly problematic around the edges of the WSI. Obviously, none of the pixels outside of the core biopsies should be labeled as positive for any of the MIF channels. Hence, rescaling the pixel distribution removes the unnecessary signal. 

Next adapative thresholding is applied. The signal on the interstism does not express equal intensity across all core biopsies, nor is it evenly expressed within a single core biopsy. To account for this, CV2 implementation of adaptive thresholding adjusts the threshold value for local regions. After experimentation, a block size of 199 was chosen for the hyperparameter for adaptive thresholding.
    
### Computing the Core Biopsy Mask

After thresholding the individual channels in the MIF, the next step in the segmention pipeline is to segment the core biopsies. This is done with the following steps:
1. Add all MIF channels used in the pipeline to a temporary mask
2. Close any holes with an area smaller than $500^2$ pixels 
3. Open any holes with an area smaller than 5000 pixels

### Computing the Interstism Mask 

Next, the interstism mask is created, by following these steps:
1. Remove the CD31 and CD34 (these are signals that are expressed in the glomeruli, which we don't want in the interstism mask) from the MIF collagen channel
2. Open (ie remove) any region smaller than $50^2$ pixels
3. Perform closing with a disk size of 6. This helps close any gaps in the interstism
4. Close any holes smaller than 500 pixels in area. This step includes the insterstisial capillaries in the interstism mask. I'm not sure if this is correct

### Computing the Tubule Mask

Finally the tubule mask is created by:
1. Combining all thresholded tubule channels together. Specifically, this is AQP1, AQP2, CK7, Calbindin, and Panck.
2. Drawing the outline of the core biopsy mask on the interstism mask. This allows for us to segment out the tubules on the edge of the core biopsies
3. For each hole in the interstism mask, check if a enough tubule markers exist in this region. If so, draw this region on the tubule mask. Currently, 50% of the region needs to have a tubule marker to be considered a tubule. 


## To Do
- add Iain as a collaborator/ how does git work with alpaca?

