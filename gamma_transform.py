import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def apply_gamma(image_array, gamma):
    normalized = image_array / 255.0
    gamma_corrected = np.clip(normalized ** gamma, 0, 1)
    return gamma_corrected

def find_peak_threshold(image_array, bins=256, threshold=100):
    hist, bin_edges = np.histogram(image_array, bins=bins)
    print(hist)

    # Traverse from bright to dark (high to low bin index)
    for i in reversed(range(bins)):
        if hist[i] >= threshold:
            return (bin_edges[i] + bin_edges[i + 1]) / 2

    # Fallback: use literal max if nothing crosses threshold
    return np.max(image_array)

def normalize_by_peak(image_array, peak_value):
    return np.clip(image_array / max(peak_value, 1e-5), 0, 1)

def process_gamma_folder(input_dir="inverse_lut_output", output_dir="gamma_output", gamma=0.5):
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    for fname in tqdm(files, desc=f"Gamma→Normalize"):
        path_in = os.path.join(input_dir, fname)
        path_out = os.path.join(output_dir, fname)

        img = Image.open(path_in).convert("L")
        arr = np.array(img).astype(np.float32)

        # peak = find_peak_threshold(arr, threshold=100)
        # arr = normalize_by_peak(arr, peak)
        arr = apply_gamma(arr, gamma)

        out_arr = (arr * 255).astype(np.uint8)
        Image.fromarray(out_arr, mode="L").save(path_out)

if __name__ == "__main__":
    process_gamma_folder(
        "inverse_lut_output",
        "gamma_output",
        gamma=2
    )
