import os
import numpy as np
from matplotlib import colormaps
from scipy.spatial import KDTree
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class InverseColormap:
    def __init__(self, colormap_name="viridis", lut_size=256):
        self.scalar_values = np.linspace(0, 1, lut_size)
        self.colormap_rgb = colormaps[colormap_name](self.scalar_values)[:, :3]
        self.tree = KDTree(self.colormap_rgb)

    def map_rgb_to_scalar(self, rgb_image):
        flat = rgb_image.reshape(-1, 3)
        _, indices = self.tree.query(flat)
        scalar_values = self.scalar_values[indices]
        return scalar_values.reshape(rgb_image.shape[:-1])


# Global LUT instance to avoid pickling
lut = None

def init_lut(colormap_name="viridis"):
    global lut
    lut = InverseColormap(colormap_name)


def process_single_file(args):
    fname, input_dir, output_dir = args
    input_path = os.path.join(input_dir, fname)
    output_path = os.path.join(output_dir, fname)

    img = Image.open(input_path).convert("RGBA")
    rgba = np.array(img) / 255.0
    rgb = rgba[..., :3]

    scalar = lut.map_rgb_to_scalar(rgb)
    gray_uint8 = (scalar * 255).astype(np.uint8)
    Image.fromarray(gray_uint8, mode="L").save(output_path)


def process_folder(input_dir="data_raw", output_dir="inverse_lut_output", colormap_name="viridis"):
    os.makedirs(output_dir, exist_ok=True)
    pngs = [f for f in os.listdir(input_dir) if f.lower().endswith(".png")]
    args_list = [(fname, input_dir, output_dir) for fname in pngs]

    with ProcessPoolExecutor(initializer=init_lut, initargs=(colormap_name,)) as executor:
        list(tqdm(executor.map(process_single_file, args_list), total=len(args_list), desc="Inverse LUT"))


if __name__ == "__main__":
    process_folder("data_cropped", "inverse_lut_output", "viridis")
