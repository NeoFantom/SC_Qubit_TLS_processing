# write a program that extracts the pixels into a set.

import numpy as np
import cv2

def extract_pixels(image):
    """
    Extracts unique pixel values from a 2D image array.
    
    Parameters:
    image (np.ndarray): A 2D numpy array representing the image.
    
    Returns:
    set: A set of unique pixel values.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array.")
    
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D array.")
    
    unique_pixels = set(image.flatten())
    return unique_pixels

if __name__ == "__main__":
    # Example usage
    sample_image = cv2.imread('stripe_mask\Q37.png', cv2.IMREAD_GRAYSCALE)
    unique_pixels = extract_pixels(sample_image)
    print("Unique pixel values:", unique_pixels)