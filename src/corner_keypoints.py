import os
import argparse
import numpy as np
import cv2  # ensure cv2 is imported properly
from src.common import list_images, read_image, to_gray, ensure_dir, write_image

def harris_corners(gray: np.ndarray,
                   block_size: int = 2,
                   ksize: int = 3,
                   k: float = 0.04,
                   thresh_ratio: float = 0.08):
    """
    Tuned Harris Corner Detector for smooth, low-texture objects.

    Changes:
      - Milder blur to preserve edge contrast
      - Lower threshold (8%) to recover weak corners
      - Smaller block size for sharper corner localization
    """
    # --- Preprocessing ---
    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0.7)  # smaller blur keeps contrast
    gray_eq = cv2.equalizeHist(gray_blur)

    # --- Harris Response ---
    gray_f = np.float32(gray_eq)
    R = cv2.cornerHarris(gray_f, block_size, ksize, k)
    R = cv2.dilate(R, None)

    # --- Normalize for visualization ---
    R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # --- Threshold ---
    T = thresh_ratio * R.max()
    nms = cv2.dilate(R, None)
    mask_nms = (R == nms) & (R > T)
    ys, xs = np.where(mask_nms)
    coords = np.stack([xs, ys], axis=1)

    return coords, R_norm


def run(in_dir: str, out_dir: str):
    ensure_dir(out_dir)
    for p in list_images(in_dir):
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        img = read_image(p)
        gray = to_gray(img)

        # Harris corner detection
        corners, R_vis = harris_corners(
            gray,
            block_size=3,   # larger neighborhood
            ksize=3,
            k=0.03,
            thresh_ratio=0.1   # increased to detect more corners
        )

        # Overlay corners on image
        overlay = img.copy()
        for (x, y) in corners:
            cv2.circle(overlay, (int(x), int(y)), 3, (0, 0, 255), 1, lineType=cv2.LINE_AA)

        # Save results
        write_image(os.path.join(out_dir, f"{stem}_harris_response.png"), R_vis)
        write_image(os.path.join(out_dir, f"{stem}_harris_corners.png"), overlay)
        print(f"[OK] {name}: {len(corners)} Harris corners detected")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/outputs/corner_keypoints")
    args = ap.parse_args()
    run(**vars(args))


def detect_corners(img):
    """Simplified corner detection for Streamlit visualization."""
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img2 = img.copy()
    img2[dst > 0.01 * dst.max()] = [0, 0, 255]
    return img2
