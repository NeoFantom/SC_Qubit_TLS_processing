# README

1. Detect boundaries of color.
1. Crop by the boundaries. Crop off some colored rows to ensure no weird things happen.
1. Apply inverse lut of the transform `viridis_r` (without `_r` so that brightness is kept).
1. Apply gamma transform.
1. Denoise by column.
1. Enhance contrast.
1. Detect stripes with `./stripe_detect_noedging.py`.