# Napari User Guide

## Overview

Napari can help us overlay the mask image (which contains only pixels of 0 or 255 values) over the raw image that we need to process.

## Packages

`pip install opencv napari`

## Steps

**Do NOT open napari directly from the command line!**  
**Do NOT open napari directly from the command line!**  
**Do NOT open napari directly from the command line!**  

1. Open the python script `edit_manually.py`.  
   - This file has two strings: `bg_dir` for background image (the raw images) and `mask_dir` for masks. Edit them according to your folder names.
   - This file helps us to read images from two directories and overlay them automatically.
   - This file helps us add some useful functions to napari to help us navigate through images.
1. Run the script.  
   You should see:  
   <img src="napari_use_guide-images/2025-0628-201006.png" alt="" width="600"></img>
1. **To overview**:
   1. Scroll your wheel to zoom in or out.
   1. Press `n` for next image, `b` for previous ("before") image.
   1. Press `g` and then input a number to go to specific image.
   1. Press `s` to manually save the mask (auto-saved when pressing `b` or `n`).
   1. Change mask's overlay opacity (see screenshot, try it!).
   1. At console, you can see the image that is shown and whether an image is saved.
1. **To edit**, use one of the tools on the top (see screenshot), which are:
   1. Erase tool (or press 1)
   1. Draw tool (or press 2)
   1. Polygon tool (or press 3)
   1. Paint tool to paint a whole closed mask label (or press 4)
   1. Pick color (or press 5)
   1. Move tool (no need, you can hold space then drag)
1. **To make your operations faster**, here are some tips:
   1. Pick color is useful.
      > Explanation (not important): The color you pick is shown beside "label" (see screenshot) and whatever you pick, the mask that it generates is always binary-valued (only 0 or 255). The color is only a visual effect to distinguish different types of labels. The color for existing labels is pure red with 50% opacity but the default color of brush is not pure red, so you can pick a labelled area to use the same color as the existing labels.  

      You can pick a not-labelled area to get transparent color, then
      - paint a polygon label un-label this polygon,
      - or make a transparent polygon to un-label a large area.
   1. **Hold `alt`** and move your mouse left or right to adjust brush size. You can also adjust brush size by the slider beside "brush size" (see screenshot).
   1. **Hold `space`** then you can drag the image while your are using brush or eraser.
   1. **Hold `v`** to hide / show the mask (label layer).