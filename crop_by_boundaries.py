import os
import json
from PIL import Image
from tqdm import tqdm

def crop_image_by_bounds(image_path, bounds, pad=10):
    """Crop the image using inclusive bounds: [top, bottom, left, right]"""
    img = Image.open(image_path)
    top, bottom, left, right = bounds
    # PIL box: (left, upper, right, lower)
    cropped = img.crop((left, top+pad, right + 1, bottom-pad + 1))
    return cropped

def crop_all_images(image_folder="data_raw", boundary_file="boundaries.json", output_folder="data_cropped"):
    os.makedirs(output_folder, exist_ok=True)

    with open(boundary_file, "r") as f:
        boundaries = json.load(f)

    for filename, bounds in tqdm(boundaries.items(), desc="Cropping images"):
        input_path = os.path.join(image_folder, filename)
        output_path = os.path.join(output_folder, filename)
        cropped_img = crop_image_by_bounds(input_path, bounds)
        cropped_img.save(output_path)

if __name__ == "__main__":
    crop_all_images(
        image_folder="data_raw",
        boundary_file="boundaries.json",
        output_folder="data_cropped"
    )
