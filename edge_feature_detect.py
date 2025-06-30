import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from PIL import Image


def preprocess_downscale(img, grow=4, gcol=4, method='average'):
    H, W = img.shape[:2]
    Hs, Ws = H // grow, W // gcol
    img = img[:Hs * grow, :Ws * gcol]

    if img.ndim == 2:
        img_blocks = img.reshape(Hs, grow, Ws, gcol).astype(np.float32)
        if method == 'average':
            return img_blocks.mean(axis=(1, 3)).astype(np.uint8)
        elif method == 'max':
            return img_blocks.max(axis=(1, 3)).astype(np.uint8)
    elif img.ndim == 3:
        C = img.shape[2]
        img_blocks = img.reshape(Hs, grow, Ws, gcol, C).astype(np.float32)
        if method == 'average':
            return img_blocks.mean(axis=(1, 3)).astype(np.uint8)
        elif method == 'max':
            return img_blocks.max(axis=(1, 3)).astype(np.uint8)
    else:
        raise ValueError("Unsupported image shape")

def postprocess_upscale(mask, original_shape):
    target_w, target_h = original_shape[1], original_shape[0]
    return cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

def compute_signed_gradient_from_array(img, blur_ksize=7, sobel_ksize=3):
    gradients = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    gradients = cv2.GaussianBlur(gradients, (blur_ksize, blur_ksize), 0)
    return gradients

def fill_stripe_gaps(gradients):
    H, W = gradients.shape
    filled = np.zeros_like(gradients, dtype=np.uint8)
    for col in range(W):
        grad_col = gradients[:, col]
        upper_idxs = np.where(grad_col > 0)[0]
        lower_idxs = np.where(grad_col < 0)[0]
        for u in upper_idxs:
            lower_below = lower_idxs[lower_idxs > u]
            if len(lower_below) == 0:
                continue
            d = lower_below[0]
            filled[u:d+1, col] = 255
    return filled

def extract_stripe_masks(gradients, grad_thres=30, coarse_w=100, fine_w=10, kernel_h=1, coarse_floor=30, fine_floor=50):
    abs_grad = np.abs(gradients)
    strong_edges = np.where(abs_grad >= grad_thres, abs_grad, 0).astype(np.uint8)

    coarse_mat = cv2.getStructuringElement(cv2.MORPH_RECT, (coarse_w, kernel_h))
    coarse_closed = cv2.morphologyEx(strong_edges, cv2.MORPH_CLOSE, coarse_mat)

    fine_mat = cv2.getStructuringElement(cv2.MORPH_RECT, (fine_w, kernel_h))
    fine_closed = cv2.morphologyEx(strong_edges, cv2.MORPH_CLOSE, fine_mat)

    merged = np.zeros_like(coarse_closed, dtype=np.uint8)
    condition = (coarse_closed > coarse_floor) & (fine_closed > fine_floor)
    merged[condition] = np.maximum(coarse_closed[condition], fine_closed[condition])

    return coarse_closed, fine_closed, merged


def filter_by_shape(mask, min_width=10, min_height=5):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if w >= min_width and h >= min_height:
            filtered[labels == i] = 255
    return filtered


def visualize_signed_gradient(gradients, scale=4.0):
    H, W = gradients.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    scaled = np.clip(gradients * scale, -255, 255)
    pos = scaled > 0
    neg = scaled < 0
    rgb[..., 1][pos] = np.uint8(scaled[pos])
    rgb[..., 2][neg] = np.uint8(-scaled[neg])
    return rgb


def overlay_mask_boundaries_on_image(image, mask, boundary_thick=1, color_mode='color'):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if color_mode == 'gray':
        if len(image.shape) == 2:
            overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            overlay = image.copy()
    else:
        if image.shape[2] == 3:
            overlay = image[..., ::-1].copy()  # RGB to BGR for OpenCV
        else:
            overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), thickness=boundary_thick)
    return overlay[..., ::-1]  # Convert back to RGB


def plot_and_save_images(images: dict, output_dir: str, figsize=(7, 2.5), cmap_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    N = len(images)
    cols = 1 if N <= 3 else 2
    rows = (N + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows))
    axs = np.atleast_1d(axs).flatten()

    for i, (title, img) in enumerate(images.items()):
        ax = axs[i]
        cmap = None
        if cmap_dict and title in cmap_dict:
            cmap = cmap_dict[title]
        elif img.ndim == 2:
            cmap = 'gray'
        display_title = f"{i+1} {title}"
        ax.imshow(img, cmap=cmap)
        ax.set_title(display_title)
        ax.axis("off")
        filename = f"{i+1}_{title.replace(' ', '_').lower()}.png"
        save_path = os.path.join(output_dir, filename)
        if img.ndim == 2:
            cv2.imwrite(save_path, img)
        else:
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr)

    for j in range(i + 1, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    qname = "Q35"
    input_path = f"gamma_output/{qname}.png"
    raw_color_path = f"data_raw/{qname}.png"
    boundaries_path = "boundaries.json"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    grow, gcol = 1, 1
    grad_thres = 50
    coarse_w = 50
    fine_w = 10
    kernel_h = 2
    coarse_floor = 30
    fine_floor = 30

    stripe_width = 5
    stripe_length = 200
    boundary_thick = 3

    with open(boundaries_path, 'r') as f:
        bounds = json.load(f)
    top, bottom, left, right = bounds[f'{qname}.png']
    pad = 10
    pil_img = Image.open(raw_color_path).convert("RGB")
    pil_crop = pil_img.crop((left, top + pad, right + 1, bottom - pad + 1))
    raw_img = np.array(pil_crop)

    original_img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    print("Loaded original image with shape:", original_img.shape)

    downscaled_img = preprocess_downscale(original_img, grow, gcol)
    print("Downscaled image shape:", downscaled_img.shape)

    # gradients = compute_signed_gradient_from_array(downscaled_img)
    # print("Computed gradient image")

    # gradient_vis = visualize_signed_gradient(gradients)
    # print("Generated gradient visualization")

    # stripe_fill = fill_stripe_gaps(gradients)
    # print("Filled gaps in stripe gradients")

    stripe_fill = downscaled_img

    coarse_mask, fine_mask, merged_mask = extract_stripe_masks(
        stripe_fill,
        grad_thres=grad_thres,
        coarse_w=coarse_w,
        fine_w=fine_w,
        kernel_h=kernel_h,
        coarse_floor=coarse_floor,
        fine_floor=fine_floor
    )
    print("Extracted stripe masks")

    filtered_mask = filter_by_shape(merged_mask, min_width=stripe_length, min_height=stripe_width)
    print("Filtered mask by connected component shape")

    mask_full = postprocess_upscale(filtered_mask, original_img.shape)
    print("Upscaled mask to original resolution")

    overlayed = overlay_mask_boundaries_on_image(raw_img, mask_full, boundary_thick=boundary_thick, color_mode='color')
    print("Generated overlay visualization")

    plot_and_save_images({
        "Raw Image": raw_img,
        # "Signed Gradient Visualization": gradient_vis,
        "Coarse Edge Mask": coarse_mask,
        "Fine Edge Mask": fine_mask,
        "Merged Stripe Mask": merged_mask,
        "Mask Full": mask_full,
        "Overlay on Original": overlayed
    }, output_dir)
    print("Saved all images")