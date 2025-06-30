import os
import cv2
from tqdm import tqdm

input_dir = "denoised"
output_dir = "contrast_enhanced"
os.makedirs(output_dir, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))

for fname in tqdm(os.listdir(input_dir), desc="Enhancing contrast"):
    if not fname.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_dir, fname)
    out_path = os.path.join(output_dir, fname)

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    enhanced = clahe.apply(img)
    cv2.imwrite(out_path, enhanced)
