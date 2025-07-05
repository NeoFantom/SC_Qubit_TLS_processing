# README

1. Detect boundaries of color. Save them in [./boundaries.json](./boundaries.json).
1. Crop by the boundaries. Crop with a padding to ensure no weird things happen at borders.
1. Apply inverse lut of the transform `viridis_r` (without `_r` so that brightness is kept).
1. Apply gamma transform.
1. Denoise by column.
1. (Optional, not applied in the final workflow.) Enhance contrast.
1. Detect stripes with [./stripe_detect_noedging.py](./stripe_detect_noedging.py), or batch process in [./stripe_batch.py](./stripe_batch.py)
1. Edit them manually by running [./edit_manually.py](./edit_manually.py), which starts a GUI of `napari`. See details in [./napari_use_guide.md](./napari_use_guide.md).
