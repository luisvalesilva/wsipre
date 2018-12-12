# -*- coding: utf-8 -*-

"""
wsipre.tissue
-------------

Functionality to detect tissue RoIs.
"""

import numpy as np
import cv2
from skimage import morphology


def otsu_filter(channel, gaussian_blur=True):
    """Otsu filter."""
    if gaussian_blur:
        channel = cv2.GaussianBlur(channel, (5, 5), 0)
    channel = channel.reshape((channel.shape[0], channel.shape[1]))

    return cv2.threshold(
        channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def detect_tissue(wsi, downsampling_factor=64):
    """Find RoIs containing tissue in WSI.

    Generate mask locating tissue in an WSI. Inspired by method used by
    Wang et al. [1]_.

    .. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew
    H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",
    arXiv:1606.05718

    Parameters
    ----------
    wsi: OpenSlide/AnnotatedOpenSlide class instance
        The whole-slide image (WSI) to detect tissue in.
    downsampling_factor: int
        The desired factor to downsample the image by, since full WSIs will
        not fit in memory. The image's closest level downsample is found
        and used.

    Returns
    -------
    Binary mask as numpy 2D array, RGB slide image (in the used
    downsampling level, in case the user is visualizing output examples)
    and downsampling factor.

    """
    # Get a downsample of the whole slide image (to fit in memory)
    downsampling_factor = min(
        wsi.level_downsamples, key=lambda x: abs(x - downsampling_factor))
    level = wsi.level_downsamples.index(downsampling_factor)

    slide = wsi.read_region((0, 0), level, wsi.level_dimensions[level])
    slide = np.array(slide)[:, :, :3]

    # Convert from RGB to HSV color space
    slide_hsv = cv2.cvtColor(slide, cv2.COLOR_BGR2HSV)

    # Compute optimal threshold values in each channel using Otsu algorithm
    _, saturation, _ = np.split(slide_hsv, 3, axis=2)

    mask = otsu_filter(saturation, gaussian_blur=True)

    # Make mask boolean
    mask = mask != 0

    mask = morphology.remove_small_holes(mask, area_threshold=5000)
    mask = morphology.remove_small_objects(mask, min_size=5000)

    mask = mask.astype(np.uint8)
    _, mask_contours, _ = cv2.findContours(
        mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return mask_contours, slide, downsampling_factor
