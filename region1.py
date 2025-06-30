import cv2
import numpy as np
from tqdm import tqdm
from skimage.segmentation import flood
from scipy.ndimage import maximum_filter


def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur


def find_local_maxima(img, threshold_min):
    neighborhood = maximum_filter(img, size=3)
    peaks = (img == neighborhood) & (img >= threshold_min)
    coords = np.argwhere(peaks)
    return coords


def extract_regions(img, threshold_min=50, brightness_tol=10, connect_tol=0.05):
    seeds = find_local_maxima(img, threshold_min)
    assigned = np.zeros(img.shape, dtype=bool)
    regions = []

    for y, x in tqdm(seeds, desc='Flood-filling from peaks'):
        if assigned[y, x]:
            continue
        mask = flood(img, (y, x), tolerance=brightness_tol)
        mask &= ~assigned

        if np.count_nonzero(mask) == 0:
            continue

        xs = np.where(mask)[1]
        if xs.size > 0:
            region_width = xs.max() - xs.min() + 1
            kernel_size = max(1, int(connect_tol * region_width))
            kernel = np.ones((1, kernel_size), np.uint8)
            closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
            mask = closed.astype(bool)

        mean_intensity = img[mask].mean()
        assigned[mask] = True
        regions.append((mean_intensity, mask))

    regions.sort(key=lambda x: x[0], reverse=True)
    return [r[1] for r in regions]


def overlay_regions(original, regions):
    overlay = original.copy().astype(np.float32)
    for mask in tqdm(regions, desc='Overlaying regions'):
        color = np.random.randint(0, 256, size=3, dtype=np.uint8)
        overlay[mask] = overlay[mask] * 0.5 + color * 0.5
    return overlay.astype(np.uint8)

if __name__ == '__main__':
    input_path = 'output/3_coarse_convolution.png'
    output_path = 'output/flood_fill.png'
    brightness_tol = 10
    connect_tol = 0.05
    threshold_ratio = 0.5

    orig = cv2.imread(input_path)
    if orig is None:
        print(f"Error: could not read '{input_path}'")
        exit(1)

    proc = preprocess_image(orig)
    th_min = int(proc.max() * threshold_ratio)
    regions = extract_regions(proc, threshold_min=th_min,
                              brightness_tol=brightness_tol,
                              connect_tol=connect_tol)
    result = overlay_regions(orig, regions)
    cv2.imwrite(output_path, result)