---
marp: true
---

# Notes about TLS_data_process.ipynb

How to detect TLS fluctuation: inner product is useful indeed, but we need to improvise a little.

Instead of dotting the array `pixel_value[position]` or $S(x)$, let's do a histogram-like operation (or inverse function) to get `position[pixel_value]` or $x(S)$, then let's see for a similar brightness pixel, how far away they appear in adjacent columns. But of course, we must first take the average of a whole band and then take each band as a whole.