import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm

def plot_part(index, dindex, x, deviation, vmin, vmax, ax, cmap="viridis"):
    """
    Plots a vertical stripe at position x with data z repeated across width dx,
    colored according to value, and placed along y-axis.
    """
    deviation = np.array([deviation, deviation]).T
    dx = x[1] - x[0]
    # extent = (left, right, bottom, top)
    extent = (index - dindex / 2, index + dindex / 2, x[0] - dx / 2, x[-1] + dx / 2)
    im = ax.imshow(
        deviation,
        extent=extent,
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        cmap=cmap
    )
    return im

if __name__ == '__main__':
    # === Configuration ===
    data_path = "./TLS_DATA_2024Nov_2025May.npy"
    output_dir = "./TLS_analyze/data_raw"
    # qubits = range(72)
    qubits = [69]
    dindex = 5
    vmin, vmax = -0.5, 0.1
    y_limits = (2.8e9, 4.6e9)
    cmap = 'viridis_r'
    figsize = (16, 8)
    dpi = 360

    # === Setup ===
    plt.close('all')
    os.makedirs(output_dir, exist_ok=True)
    TLS_DATA = np.load(data_path, allow_pickle=True).item()

    for qubit in qubits:
        print(f"Processing Q{qubit}...")
        fig, ax = plt.subplots(figsize=figsize)

        xs = TLS_DATA[qubit]["xs"]
        ys = TLS_DATA[qubit]["ys"]

        times = TLS_DATA[qubit]["times"]
        # Convert time strings to date objects
        dates = [datetime.strptime(t, '%a %b %d %H:%M:%S %Y').date() for t in times]
        unique_dates, unique_inds = np.unique(dates, return_index=True) # type: ignore

        # Plot each vertical stripe
        for i, (x, y) in tqdm(enumerate(zip(xs, ys))):
            if i % 5 != 0:
                pass
            deviation = y - np.median(y)
            plot_part(index=i, dindex=dindex, x=x, deviation=deviation, vmin=vmin, vmax=vmax, ax=ax, cmap=cmap)
            

        ax.autoscale_view(scaley=False)
        ax.set_title(f'Q{qubit}')
        ax.set_xticks(unique_inds)
        ax.set_xticklabels([dt.strftime('%y-%m-%d') for dt in unique_dates])
        ax.xaxis.set_tick_params(rotation=-90, labelsize=10)
        ax.set_ylim(*y_limits)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'Q{qubit}.png')
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.show()

        break
        plt.close(fig)

