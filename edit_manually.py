import napari
import numpy as np
import cv2
import imageio.v2 as imageio
import os
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
This script provides an interactive interface for manually editing binary stripe masks
using Napari.

Background images are loaded from the 'data_cropped' directory, and corresponding
binary mask images are loaded from the 'stripe_mask' directory. The filenames are
sorted alphanumerically to ensure consistent navigation.

Functionality provided:
- Display the background image and editable overlay mask.
- Press 's' to save the current edited mask to disk.
- Press 'n' to save and move to the next image.
- Press 'b' to save and move to the previous image (overrides default Napari shortcut).
- Press 'g' then input a 0-based index and jump to that image.
- Edited masks are saved as 8-bit PNG files (with pixel = 0 or 255).

All operations preserve the current editing state and allow cycling through
the image list in a non-destructive manner.
"""


bg_dir = 'data_cropped'
mask_dir = 'stripe_mask_manual'

def natural_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

filenames = sorted(
    [f for f in os.listdir(mask_dir) if f.endswith('.png')],
    key=natural_key
)

reference_pointer = [0]  # use list for mutability in closures

def load_images(i):
    fname = filenames[i]
    bg_path = os.path.join(bg_dir, fname)
    mask_path = os.path.join(mask_dir, fname)

    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    bg = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)

    mask = imageio.imread(mask_path)
    if mask.ndim == 3:
        mask = mask[..., 0]

    return fname, bg, mask

fname, bg, mask = load_images(reference_pointer[0])

viewer = napari.view_image(bg, name='base')
label_layer = viewer.add_labels(mask, name='editable_overlay', opacity=0.5)

@viewer.bind_key('s')
def save_mask(viewer):
    binary = (np.array(label_layer.data) > 0).astype(np.uint8) * 255
    out_path = os.path.join(mask_dir, filenames[reference_pointer[0]])
    imageio.imwrite(out_path, binary)
    logging.info(f"Saved: {out_path}")

@viewer.bind_key('n')
def next_image(viewer):
    save_mask(viewer)
    reference_pointer[0] = (reference_pointer[0] + 1) % len(filenames)
    fname, bg, mask = load_images(reference_pointer[0])
    viewer.layers['base'].data = bg
    label_layer.data = mask # type: ignore
    logging.info(f"Now editing: {fname}")

from napari.layers.labels import Labels
Labels.bind_key('b', None)  # Disable built-in 'b' shortcut
@viewer.bind_key('b')
def prev_image(viewer):
    save_mask(viewer)
    reference_pointer[0] = (reference_pointer[0] - 1) % len(filenames)
    fname, bg, mask = load_images(reference_pointer[0])
    viewer.layers['base'].data = bg
    label_layer.data = mask # type: ignore
    logging.info(f"Now editing: {fname}")

from qtpy.QtWidgets import QInputDialog

@viewer.bind_key('g')
def go_to_index(viewer):
    i, ok = QInputDialog.getInt(
        None, "Go to Image", "Enter 0-based image index:",
        value=reference_pointer[0], min=0, max=len(filenames) - 1
    ) # type: ignore
    if ok:
        save_mask(viewer)
        reference_pointer[0] = i
        fname, bg, mask = load_images(i)
        viewer.layers['base'].data = bg
        label_layer.data = mask # type: ignore
        logging.info(f"Jumped to: {fname}")


napari.run()
