import os
import cv2
import numpy as np
import json
from PIL import Image
from plot_and_save_images import plot_and_save_images
from denoise import estimate_column_noise

def preprocess_downscale(img, grow=4, gcol=4, method='average'):
    H, W = img.shape[:2]
    Hs, Ws = H // grow, W // gcol
    img = img[:Hs * grow, :Ws * gcol]
    if img.ndim == 2:
        img_blocks = img.reshape(Hs, grow, Ws, gcol).astype(np.float32)
        return img_blocks.mean(axis=(1, 3)).astype(np.uint8) if method == 'average' else img_blocks.max(axis=(1, 3)).astype(np.uint8)
    elif img.ndim == 3:
        C = img.shape[2]
        img_blocks = img.reshape(Hs, grow, Ws, gcol, C).astype(np.float32)
        return img_blocks.mean(axis=(1, 3)).astype(np.uint8) if method == 'average' else img_blocks.max(axis=(1, 3)).astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape")

def postprocess_upscale(mask, original_shape):
    return cv2.resize(mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)

def extract_stripe_conv_multiple(img, kernel_sizes, min_brightness=30):
    abs_grad = np.abs(img)
    strong_stripes = np.where(abs_grad >= min_brightness, abs_grad, 0).astype(np.float32)
    result = np.zeros_like(strong_stripes, dtype=np.float32)
    K = 0.8  # Ratio to keep the leading kernels' results
    for w, h in kernel_sizes[::-1]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w, h))
        closed = cv2.morphologyEx(strong_stripes, cv2.MORPH_CLOSE, kernel)
        result = (result * (1-K) + np.asarray(closed) * K)
    
    # normalize the result to the range [0, 255]
    result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX) # type: ignore
    return result.astype(np.uint8)

def filter_by_shape(mask, min_width=10, min_height=5):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        x, y, w, h, _ = stats[i]
        if w >= min_width and h >= min_height:
            filtered[labels == i] = 255
    return filtered

def overlay_mask_boundaries_on_image(image, mask, boundary_thick=1, color_mode='color'):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if (color_mode == 'gray' and image.ndim == 2) else image.copy()
    if color_mode == 'color' and image.shape[2] == 3:
        overlay = image[..., ::-1].copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=boundary_thick)
    return overlay[..., ::-1]

if __name__ == "__main__":
    qname = "Q69"
    grow, gcol = 1, 1
    min_brightness = 10
    kernel_list = [(100, 2), (50, 2), (20, 2)]
    stripe_width_min, stripe_length_min = 2, 200
    boundary_thick = 3  # Thickness of the boundary lines
    pad = 10  # Padding for cropping

    input_path = f"denoised/{qname}.png"
    raw_path = f"data_raw/{qname}.png"
    boundaries_path = "boundaries.json"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    with open(boundaries_path, 'r') as f:
        top, bottom, left, right = json.load(f)[f'{qname}.png']
    crop_box = (left, top + pad, right + 1, bottom - pad + 1)
    raw_img = np.array(Image.open(raw_path).convert("RGB").crop(crop_box))

    original_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    img_mean = np.mean(original_img, axis=1)
    img_mean = np.abs(np.gradient(np.gradient(img_mean)))
    img_mean = cv2.GaussianBlur(img_mean, (5, 1), 0)
    def plot_mean(ax):
        ax.plot(img_mean[::-1], list(range(len(img_mean))))
        ax.set_title("Mean Intensity per Row")
        ax.grid(True)
    
    img_mean = img_mean / np.max(img_mean)
    # make img float array
    original_img = original_img.astype(np.float32)
    highlighted_rows = original_img * np.power(img_mean, 1/5)
    highlighted_rows = highlighted_rows.astype(np.uint8)

    downscaled = preprocess_downscale(highlighted_rows, grow=grow, gcol=gcol)

    coarse_conv = extract_stripe_conv_multiple(downscaled, kernel_list, min_brightness=min_brightness)
    denoised = estimate_column_noise(
        coarse_conv, low_cut=0, high_cut=1/10,
        high_check=1/20, high_ratio_max=0.5, low_ratio_max=1)

    filtered = filter_by_shape(denoised, min_width=stripe_length_min, min_height=stripe_width_min)
    mask_full = postprocess_upscale(filtered, original_img.shape)
    overlayed = overlay_mask_boundaries_on_image(raw_img, mask_full, boundary_thick=boundary_thick, color_mode='color')

    plot_and_save_images({
        "Raw Image": raw_img,
        "Highlighted Rows": highlighted_rows,
        "Coarse Convolution": coarse_conv,
        "Denoised Columns": denoised,
        "mask_full": mask_full,
        "Overlay on Original": overlayed
    }, output_dir)
