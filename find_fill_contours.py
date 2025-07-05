import cv2
import numpy as np
import os
from tqdm import tqdm

def compute_edge_strength(img):
    # Compute gradient magnitude
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    edge = np.sqrt(grad_x**2 + grad_y**2)
    return edge

def find_and_fill_contours(img, edge, edge_thresh=0.2):
    edge_mask = (edge > edge_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_img = np.zeros_like(img, dtype=np.float32)

    for cnt in tqdm(contours):
        mask = np.zeros_like(img, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, color=1, thickness=-1)

        edge_strength = edge[mask.astype(bool)].mean()
        filled_img[mask == 1] = edge_strength

    return filled_img

def normalize_img(img):
    img = img - np.min(img)
    img = img / np.max(img) if np.max(img) > 0 else img
    return (img * 255).astype(np.uint8)

if __name__ == '__main__':
    # Change the path to your image
    img_path = 'denoised/Q0.png'
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"{img_path} not found")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = img.astype(np.float32) / 255.0

    edge = compute_edge_strength(img)
    filled = find_and_fill_contours(img, edge, edge_thresh=0.2)

    edge_vis = normalize_img(edge)
    filled_vis = normalize_img(filled)

    cv2.imshow('Original', normalize_img(img))
    cv2.imshow('Edge Strength', edge_vis)
    cv2.imshow('Filled Bright Regions', filled_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
