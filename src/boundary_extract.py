#!/usr/bin/env python3
# -------------------------------------------------------------
# boundary_extract.py
# Exact object boundary extraction using classical CV methods.
# -------------------------------------------------------------
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import argparse
import numpy as np
import cv2  # use cv2 consistently
from src.common import list_images, read_image, to_gray, ensure_dir, write_image


def extract_boundary(
    img,
    blur_size=5,
    canny1=60,
    canny2=120,
    min_area=2000
):
    """
    Classical boundary extraction with preprocessing and
    geometric contour filtering (for rectangular or near-rectangular objects).
    """
    # --- 1. Grayscale & contrast normalization ---
    gray = to_gray(img)
    gray_eq = cv2.equalizeHist(gray)

    # --- 2. Edge sharpening ---
    lap = cv2.Laplacian(gray_eq, cv2.CV_8U)
    sharp = cv2.addWeighted(gray_eq, 1.5, lap, -0.5, 0)

    # --- 3. Smooth to reduce noise ---
    blur = cv2.GaussianBlur(sharp, (blur_size, blur_size), 0)

    # --- 4. Canny edge detection ---
    edges = cv2.Canny(blur, canny1, canny2)

    # --- 5. Morphological closing to connect fragmented edges ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # --- 6. Find contours ---
    contours, _ = cv2.findContours(edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, edges_closed, None

    # --- 7. Filter contours: find most rectangular, largest contour ---
    h, w = gray.shape
    best_contour = None
    best_score = 0

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        x, y, cw, ch = cv2.boundingRect(c)
        aspect = cw / float(ch)
        rect_area = cw * ch
        fill_ratio = area / float(rect_area + 1e-6)

        # composite score: large, rectangular, good fill
        score = fill_ratio * (area / (h * w)) * (1.0 - abs(aspect - 1.4))
        if score > best_score:
            best_score = score
            best_contour = c

    # fallback: largest contour if scoring fails
    if best_contour is None:
        best_contour = max(contours, key=cv2.contourArea)

    # --- 8. Draw overlay ---
    overlay = img.copy()
    cv2.drawContours(overlay, [best_contour], -1, (0, 0, 255), 3)

    # --- 9. Create binary mask ---
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, -1)  # filled mask

    return overlay, edges_closed, mask


def run(in_dir, out_dir):
    ensure_dir(out_dir)

    for p in list_images(in_dir):
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        img = read_image(p)

        overlay, edges, mask = extract_boundary(img)
        if overlay is not None:
            write_image(os.path.join(out_dir, f"{stem}_edges.png"), edges)
            write_image(os.path.join(out_dir, f"{stem}_boundary.png"), overlay)
            write_image(os.path.join(out_dir, f"{stem}_mask.png"), mask)
            print(f"[OK] {name}: boundary extracted, mask saved")
        else:
            print(f"[WARN] {name}: no valid contour found")


# -------------------------------------------------------------
# Simple version for Streamlit app real-time use
# -------------------------------------------------------------
def extract_boundary_simple(img):
    """Simplified boundary extraction for live visualization (Streamlit)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = img.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw", help="input image directory")
    ap.add_argument("--out_dir", default="data/outputs/boundary_extract", help="output directory")
    args = ap.parse_args()
    run(**vars(args))
