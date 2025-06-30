import os
import json
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from stripe_detect_noedging import (
    preprocess_downscale,
    extract_stripe_conv_multiple,
    estimate_column_noise,
    filter_by_shape,
    postprocess_upscale,
    overlay_mask_boundaries_on_image,
)

# Parameters
grow, gcol = 1, 1
min_brightness = 30
kernel_list = [(100, 2), (50, 2), (20, 2)]
stripe_width_min, stripe_length_min = 2, 200
boundary_thick = 3
pad = 10

input_dir = "denoised"
raw_dir = "data_raw"
json_path = "boundaries.json"
conv_dir = "stripe_conv"
mask_dir = "stripe_mask"
overlay_dir = "stripe_overlay"
tobeoverlayed_dir = "stripe_tobeoverlayed"

os.makedirs(conv_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(overlay_dir, exist_ok=True)
os.makedirs(tobeoverlayed_dir, exist_ok=True)

with open(json_path, 'r') as f:
    boundaries = json.load(f)

def process_one(fname):
    if not fname.lower().endswith(".png"):
        return
    qname = os.path.splitext(fname)[0]
    if f"{qname}.png" not in boundaries:
        return

    top, bottom, left, right = boundaries[f"{qname}.png"]
    crop_box = (left, top + pad, right + 1, bottom - pad + 1)

    raw_path = os.path.join(raw_dir, f"{qname}.png")
    raw_img = np.array(Image.open(raw_path).convert("RGB").crop(crop_box))
    input_img = cv2.imread(os.path.join(input_dir, fname), cv2.IMREAD_GRAYSCALE)
    if input_img is None:
        return

    img_mean = np.mean(input_img, axis=1)
    img_mean = np.abs(np.gradient(np.gradient(img_mean)))
    img_mean = cv2.GaussianBlur(img_mean, (5, 1), 0)
    img_mean = img_mean / np.max(img_mean)
    highlighted_rows = (input_img.astype(np.float32) * np.power(img_mean, 1 / 5)).astype(np.uint8)

    downscaled = preprocess_downscale(highlighted_rows, grow=grow, gcol=gcol)
    coarse_conv = extract_stripe_conv_multiple(downscaled, kernel_list, min_brightness=min_brightness)
    denoised = estimate_column_noise(
        coarse_conv, low_cut=0, high_cut=1/10,
        high_check=1/20, high_ratio_max=0.5, low_ratio_max=1
    )

    filtered = filter_by_shape(denoised, min_width=stripe_length_min, min_height=stripe_width_min)
    mask_full = postprocess_upscale(filtered, input_img.shape)
    overlayed = overlay_mask_boundaries_on_image(raw_img, mask_full, boundary_thick=boundary_thick, color_mode='color')

    denoised = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(conv_dir, f"{qname}.png"), denoised)

    mask_full = cv2.cvtColor(mask_full, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(os.path.join(mask_dir, f"{qname}.png"), mask_full)

    overlayed = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(overlay_dir, f"{qname}.png"), overlayed)

    tobeoverlayed = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(tobeoverlayed_dir, f"{qname}.png"), tobeoverlayed)

if __name__ == "__main__":
    fnames = os.listdir(input_dir)
    with Pool(cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_one, fnames), total=len(fnames), desc="Processing"))
