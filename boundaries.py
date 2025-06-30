import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

def is_bw(pixel, tol=3):
    r, g, b = [int(c) for c in pixel]  # prevent uint8 overflow
    return abs(r - g) < tol and abs(r - b) < tol and abs(g - b) < tol

def find_bounds_central_lines(arr, mode='bw', tol=3, color_thresh=30):
    H, W, _ = arr.shape
    mid_row = arr[H // 2, :, :]
    mid_col = arr[:, W // 2, :]

    if mode == 'bw':
        row_mask = np.array([not is_bw(px, tol) for px in mid_row])
        col_mask = np.array([not is_bw(px, tol) for px in mid_col])
    elif mode == 'chroma':
        row_mask = np.std(mid_row, axis=1) > color_thresh
        col_mask = np.std(mid_col, axis=1) > color_thresh
    else:
        raise ValueError("mode must be 'bw' or 'chroma'")

    if not np.any(row_mask) or not np.any(col_mask):
        return None

    left = np.argmax(row_mask)
    right = W - 1 - np.argmax(row_mask[::-1])
    top = np.argmax(col_mask)
    bottom = H - 1 - np.argmax(col_mask[::-1])
    return [int(top), int(bottom), int(left), int(right)]

def process_all_images(folder=".", output_file="boundaries.json", mode='bw'):
    boundaries = {}
    pngs = [f for f in os.listdir(folder) if f.lower().endswith(".png")]
    for filename in tqdm(pngs, desc="Processing images"):
        path = os.path.join(folder, filename)
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        bounds = find_bounds_central_lines(arr, mode=mode)
        if bounds:
            boundaries[filename] = bounds

    with open(output_file, "w") as f:
        json.dump(boundaries, f, indent=2)

if __name__ == "__main__":
    process_all_images(folder="data_raw", output_file="boundaries.json", mode='bw')
