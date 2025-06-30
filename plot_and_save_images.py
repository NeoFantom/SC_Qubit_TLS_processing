import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt

def plot_and_save_images(images: dict, output_dir: str, figsize=(7, 2.5), cmap_dict=None):
    os.makedirs(output_dir, exist_ok=True)
    N = len(images)
    cols = 1 if N <= 3 else 2
    rows = (N + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(figsize[0]*cols, figsize[1]*rows))
    axs = np.atleast_1d(axs).flatten()

    for i, (title, plot) in enumerate(images.items()):
        ax = axs[i]
        display_title = f"{i+1} {title}"
        filename = f"{i+1}_{title.replace(' ', '_').lower()}.png"
        save_path = os.path.join(output_dir, filename)

        try:
            # Case 1: function callback
            if callable(plot):
                plot(ax)  # user-defined callback for complex plotting
                ax.set_title(display_title)
                fig.savefig(save_path)

            # Case 2: (x, y) curve
            elif isinstance(plot, tuple) and len(plot) == 2 and all(isinstance(arr, np.ndarray) for arr in plot):
                x, y = plot
                ax.plot(x, y)
                ax.set_title(display_title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.grid(True)
                fig.savefig(save_path)

            # Case 3: image
            elif isinstance(plot, np.ndarray) and plot.ndim in [2, 3]:
                cmap = None
                if cmap_dict and title in cmap_dict:
                    cmap = cmap_dict[title]
                elif plot.ndim == 2:
                    cmap = 'gray'
                ax.imshow(plot, cmap=cmap)
                ax.set_title(display_title)
                ax.axis("off")
                if plot.ndim == 2:
                    bgr = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, bgr)
                else:
                    bgr = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(save_path, bgr)
        except Exception as e:
            print(f"Error processing {title}: {e}")
            raise e

    for j in range(i + 1, len(axs)): # type: ignore
        axs[j].axis("off")

    plt.tight_layout()
    plt.show()
