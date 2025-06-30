import os
import numpy as np
from matplotlib import colormaps
from scipy.spatial import KDTree
from PIL import Image
from tqdm import tqdm

class InverseColormap:
    def __init__(self, colormap_name, lut_size=256):
        self.scalar_values = np.linspace(0, 1, lut_size)
        self.colormap_rgb = colormaps[colormap_name](self.scalar_values)[:, :3]  # drop alpha
        self.tree = KDTree(self.colormap_rgb)

    def map_rgb_to_scalar(self, rgb_image):
        flat = rgb_image.reshape(-1, 3)
        _, indices = self.tree.query(flat)
        scalar_values = self.scalar_values[indices]
        return scalar_values.reshape(rgb_image.shape[:-1])

def process_folder(input_dir="data_raw", output_dir="inverse_lut_output", colormap_name="viridis"):
    os.makedirs(output_dir, exist_ok=True)
    lut = InverseColormap(colormap_name=colormap_name)

    pngs = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    for fname in tqdm(pngs, desc="Inverse LUT"):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        img = Image.open(input_path).convert("RGBA")
        rgba = np.array(img) / 255.0
        rgb = rgba[..., :3]

        scalar = lut.map_rgb_to_scalar(rgb)
        gray_uint8 = (scalar * 255).astype(np.uint8)
        Image.fromarray(gray_uint8, mode="L").save(output_path)

if __name__ == "__main__":
    process_folder(
        "data_cropped",
        "inverse_lut_output",
        "viridis"
    )
