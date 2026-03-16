"""
feature.py – Feature detection and description.

Supports:
  - ORB  (fast, binary descriptors, royalty-free)
  - SIFT (more accurate, requires opencv-contrib or opencv>=4.4 non-free disabled)

Usage:
    from stitching.feature import detect_and_describe
    kp, des = detect_and_describe(image, method='ORB')
"""

import cv2
import numpy as np


def detect_and_describe(
    image: np.ndarray,
    method: str = "ORB",
    n_features: int = 2000,
) -> tuple:
    """
    Detect keypoints and compute descriptors for a single image.

    Parameters
    ----------
    image     : BGR or grayscale numpy array.
    method    : 'ORB' (default) or 'SIFT'.
    n_features: Max number of features to retain.

    Returns
    -------
    keypoints   : list of cv2.KeyPoint
    descriptors : np.ndarray of shape (N, D)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    method = method.upper()

    if method == "ORB":
        detector = cv2.ORB_create(nfeatures=n_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)

    elif method == "SIFT":
        try:
            detector = cv2.SIFT_create(nfeatures=n_features)
        except AttributeError:
            # Older OpenCV may expose it under xfeatures2d
            detector = cv2.xfeatures2d.SIFT_create(nfeatures=n_features)
        keypoints, descriptors = detector.detectAndCompute(gray, None)

    else:
        raise ValueError(f"Unsupported method '{method}'. Choose 'ORB' or 'SIFT'.")

    if not keypoints or descriptors is None:
        raise RuntimeError(
            f"No features found with method='{method}'. "
            "Ensure the image has sufficient texture/contrast."
        )

    return keypoints, descriptors
