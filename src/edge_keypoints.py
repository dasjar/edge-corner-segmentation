import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import os, argparse
import numpy as np
import cv2 as cv
from src.common import list_images, read_image, to_gray, ensure_dir, write_image, normalize_8u

def gradient_edge_keypoints(gray: np.ndarray, thresh_ratio: float = 0.25):
    """
    Simple edge detection using the first derivative (gradient magnitude).
    Based on lecture concept: |∇I| = sqrt((dI/dx)^2 + (dI/dy)^2)

    Steps:
      1. Compute image gradients using Sobel operators.
      2. Compute gradient magnitude.
      3. Threshold strong gradients as edges.
    """
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    T = thresh_ratio * mag.max()
    mask = (mag > T).astype(np.uint8) * 255
    return mask, mag

def run(in_dir: str, out_dir: str, thresh_ratio: float):
    ensure_dir(out_dir)
    for p in list_images(in_dir):
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        img = read_image(p)
        gray = to_gray(img)

        # Detect edges
        mask, mag = gradient_edge_keypoints(gray, thresh_ratio)

        # Overlay keypoints (optional visualization)
        overlay = img.copy()
        ys, xs = np.where(mask > 0)
        for (x, y) in zip(xs, ys):
            cv.circle(overlay, (int(x), int(y)), 1, (0,255,255), -1, lineType=cv.LINE_AA)

        # Save outputs
        write_image(os.path.join(out_dir, f"{stem}_grad_edges_mask.png"), mask)
        write_image(os.path.join(out_dir, f"{stem}_grad_edges_overlay.png"), overlay)
        print(f"[OK] {name} → {len(xs)} edge keypoints")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/outputs/edge_keypoints")
    ap.add_argument("--thresh_ratio", type=float, default=0.25)
    args = ap.parse_args()
    run(**vars(args))

def detect_edges(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return cv.Canny(gray, 100, 200)
