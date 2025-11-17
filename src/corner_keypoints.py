#!/usr/bin/env python3
# -------------------------------------------------------------
# corner_keypoints.py
# Shi–Tomasi Corner Detection (clean, stable)
# -------------------------------------------------------------

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import numpy as np
from src.common import list_images, read_image, to_gray, ensure_dir, write_image


# =============================================================
# SHI–TOMASI CORNER DETECTOR (Minimum Eigenvalue)
# =============================================================
def shi_tomasi_corners(gray, 
                       max_corners=200,
                       quality=0.015,      # lower → more corners
                       min_distance=25):   # increase → fewer corners
    """
    Clean Shi–Tomasi corner detector.
    Produces stable & non-noisy corners suitable for smooth/rounded objects.
    """

    # 1. LIGHT BLUR (Shi–Tomasi doesn't need heavy blur)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 1)

    # 2. Shi–Tomasi detector
    corners = cv2.goodFeaturesToTrack(
        gray_blur,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False
    )

    if corners is None:
        return np.zeros((0, 2)), np.zeros((0, 2))

    # Convert to float
    corners = np.squeeze(corners).astype(np.float32)

    # 3. Subpixel refinement (optional but increases accuracy)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.01)
    refined = cv2.cornerSubPix(gray_blur, corners, (7,7), (-1,-1), criteria)

    return corners, refined


# =============================================================
# BATCH MODE (command-line execution)
# =============================================================
def run(in_dir: str, out_dir: str):
    ensure_dir(out_dir)

    for p in list_images(in_dir):
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]

        img = read_image(p)
        gray = to_gray(img)

        coarse, refined = shi_tomasi_corners(gray)

        # Draw results
        overlay = img.copy()

        # BIG GREEN DOTS
        for (x, y) in refined:
            cv2.circle(overlay, (int(x), int(y)), 10, (0,255,0), -1)   

        write_image(os.path.join(out_dir, f"{stem}_corners.png"), overlay)
        print(f"[OK] {name}: {len(refined)} Shi–Tomasi corners detected")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw", help="input image folder")
    ap.add_argument("--out_dir", default="data/outputs/corner_keypoints", help="output folder")
    args = ap.parse_args()
    run(**vars(args))


# =============================================================
# STREAMLIT WRAPPER (for the Web App)
# =============================================================
def detect_corners(img):
    """
    Called from Streamlit. Returns an image with bold green corners.
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coarse, refined = shi_tomasi_corners(gray)

    result = img.copy()

    for (x, y) in refined:
        cv2.circle(result, (int(x), int(y)), 10, (0,255,0), -1)

    return result
