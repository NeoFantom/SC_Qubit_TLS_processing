import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def estimate_column_noise(
        img,
        low_cut=None, high_cut=1/10,
        high_check=1/20, low_check=None,
        high_ratio_max=0.3, low_ratio_max=0.5,
        fill=True
    ):
    """
    Performs column-wise denoising by combining MAD thresholding and FFT-based frequency filtering.

    This function processes each column of the input image independently.
    It first suppresses low-intensity noise using the median absolute deviation (MAD),
    then analyzes the frequency content via FFT.
    Columns are considered noisy and completely suppressed (set to zero) if:

    1. Too much of their energy is concentrated in high-frequency components (above `high_check`).
    2. Too much energy lies in low-frequency components (below `low_check`).

    Only columns passing these checks are reconstructed using a band-pass filter defined by
    `low_cut` and `high_cut`. The result is clipped to 0–255 and returned as a denoised uint8 image.

    If `fill=True`, it linearly interpolates across fully black column regions (not at image edges).
    """
    H, W = img.shape
    result = np.zeros_like(img, dtype=np.float32)
    if low_cut is None:
        low_cut = 1 / (H * 2)
    if low_check is None:
        low_check = low_cut
    if high_check is None:
        high_check = high_cut

    for col in range(W):
        col_data = img[:, col].astype(np.float32)

        med = np.median(col_data)
        mad_thresh = np.median(np.abs(col_data - med))
        col_data[col_data < mad_thresh] = 0

        fft = np.fft.fft(col_data)
        freqs = np.fft.fftfreq(H)

        power_spectrum = np.abs(fft)**2
        total_energy = power_spectrum.sum()

        high_freq_energy = power_spectrum[np.abs(freqs) >= high_check].sum()
        low_freq_energy = power_spectrum[np.abs(freqs) <= low_check].sum()

        high_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
        low_ratio = low_freq_energy / total_energy if total_energy > 0 else 0

        if high_ratio > high_ratio_max or low_ratio > low_ratio_max:
            result[:, col] = 0
            continue

        keep = (np.abs(freqs) > low_cut) & (np.abs(freqs) < high_cut)
        fft_filtered = np.zeros_like(fft)
        fft_filtered[keep] = fft[keep]
        col_filtered = np.fft.ifft(fft_filtered).real

        result[:, col] = col_filtered

    result = np.clip(result, 0, 255).astype(np.uint8)

    if fill:
        is_black = np.all(result == 0, axis=0)
        i = 0
        while i < W:
            if not is_black[i]:
                i += 1
                continue
            start = i
            while i < W and is_black[i]:
                i += 1
            end = i - 1
            if start > 0 and end < W - 1:
                left_col = result[:, start - 1].astype(np.float32)
                right_col = result[:, end + 1].astype(np.float32)
                for k in range(start, end + 1):
                    alpha = (k - start + 1) / (end - start + 2)
                    result[:, k] = ((1 - alpha) * left_col + alpha * right_col).astype(np.uint8)

    return result

def process_file(args):
    fname, input_dir, output_dir = args
    in_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    img = cv2.imread(in_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    denoised = estimate_column_noise(img, fill=True)
    cv2.imwrite(out_path, denoised)

if __name__ == "__main__":
    input_dir: str = "gamma_output"
    output_dir: str = "denoised"
    os.makedirs(output_dir, exist_ok=True)

    fnames = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]

    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, [(f, input_dir, output_dir) for f in fnames]),
                  total=len(fnames), desc="Processing images"))
