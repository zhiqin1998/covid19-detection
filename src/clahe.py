# references from https://github.com/asalmada/x-ray-images-enhancement
from collections import OrderedDict

import imageio
import numpy as np


def run_clahe(img_path, window_size=100, clip_limit=150, n_iter=1):
    image = imageio.imread(img_path)

    if len(image.shape) > 2:
        image = to_grayscale(image)

    normalized_image = normalize(np.min(image), np.max(image), 0, 255, image)

    equalized_image = clahe(normalized_image, window_size, clip_limit, n_iter)

    return equalized_image


def clahe(image, window_size, clip_limit, n_iter):
    """Applies the CLAHE algorithm in an image.
    Parameters:
        image: image to be processed.
        window_size: size of the rectangular region around the pixel to be processed.
        clip_limit: maximum number of occurrences of the pixel in the histogram.
        n_iter: number of clipping iterations.
    Returns a processed image.
    """

    border = window_size // 2

    padded_image = np.pad(image, border, "reflect")
    shape = padded_image.shape
    padded_equalized_image = np.zeros(shape).astype(np.uint8)

    for i in range(border, shape[0] - border):
        for j in range(border, shape[1] - border):
            # Region to extract the histogram
            region = padded_image[i - border:i + border + 1, j - border:j + border + 1]
            # Calculating the histogram from region
            hist, bins = histogram(region)
            # Clipping the histogram
            clipped_hist = clip_histogram(hist, bins, clip_limit)
            # Trying to reduce the values above clipping
            for _ in range(n_iter):
                clipped_hist = clip_histogram(hist, bins, clip_limit)
            # Calculating the CDF
            cdf = calculate_cdf(hist, bins)
            # Changing the value of the image to the result from the CDF for the given pixel
            padded_equalized_image[i][j] = cdf[padded_image[i][j]]

    # Removing the padding from the image
    equalized_image = padded_equalized_image[border:shape[0] - border, border:shape[1] - border].astype(np.uint8)

    return equalized_image


def normalize(min_old, max_old, min_new, max_new, val):
    """Normalizes values to the interval [min_new, max_new]
    Parameters:
        min_old: min value from old base.
        max_old: max value from old base.
        min_new: min value from new base.
        max_new: max value from new base.
        val: float or array-like value to be normalized.
    """

    ratio = (val - min_old) / (max_old - min_old)
    normalized = (max_new - min_new) * ratio + min_new
    return normalized.astype(np.uint8)


def histogram(data):
    """Generates the histogram for the given data.
    Parameters:
        data: data to make the histogram.
    Returns: histogram, bins.
    """

    pixels, count = np.unique(data, return_counts=True)
    hist = OrderedDict()

    for i in range(len(pixels)):
        hist[pixels[i]] = count[i]

    return np.array(list(hist.values())), np.array(list(hist.keys()))


def to_grayscale(image):
    red_v = image[:, :, 0] * 0.299
    green_v = image[:, :, 1] * 0.587
    blue_v = image[:, :, 2] * 0.144
    image = red_v + green_v + blue_v

    return image.astype(np.uint8)


def clip_histogram(hist, bins, clip_limit):
    """Clips the given histogram.
    Parameters:
        hist: frequencies of each pixel.
        bins: pixels.
        clip_limit: limit to pixel frequencies.
    Returns the clipped hist.
    """

    n_bins = len(bins)

    # Removing values above clip_limit
    excess = 0
    for i in range(n_bins):
        if hist[i] > clip_limit:
            excess += hist[i] - clip_limit
            hist[i] = clip_limit

    # Redistributing exceding values
    # Calculating the values to be put on all bins
    for_each_bin = excess // n_bins
    # Calculating the values left
    leftover = excess % n_bins

    hist += for_each_bin
    for i in range(leftover):
        hist[i] += 1

    return hist


def calculate_cdf(hist, bins):
    """Calculates the normalized CDF (Cumulative Distribution Function)
    for the histogram.
    Parameters:
        hist: frequencies of each pixel.
        bins: pixels.
    Returns the CDF in a dictionary.
    """

    # Calculating probability for each pixel
    pixel_probability = hist / hist.sum()
    # Calculating the CDF (Cumulative Distribution Function)
    cdf = np.cumsum(pixel_probability)

    cdf_normalized = cdf * 255

    hist_eq = {}
    for i in range(len(cdf)):
        hist_eq[bins[i]] = int(cdf_normalized[i])

    return hist_eq
