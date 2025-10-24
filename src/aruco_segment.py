#!/usr/bin/env python3
# -------------------------------------------------------------
# aruco_segment.py
# ArUco-based segmentation for a non-rectangular object
# -------------------------------------------------------------
import os
import cv2
import numpy as np
import argparse
from src.common import list_images, ensure_dir

def detect_aruco(gray, dict_name="DICT_4X4_50"):
    """Detect ArUco markers in a grayscale image."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids

def corners_to_mask(img_shape, corners, hull_mode="convex"):
    """Convert detected marker corners to a polygon mask."""
    if len(corners) == 0:
        return None, None

    pts = np.concatenate(corners).reshape(-1, 2).astype(np.float32)
    hull = cv2.convexHull(pts)

    if hull_mode == "poly":
        peri = cv2.arcLength(hull, True)
        hull = cv2.approxPolyDP(hull, 0.01 * peri, True)

    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
    return mask, hull

def process_image(img_path, out_dir, dict_name="DICT_4X4_50"):
    """Run full pipeline for one image."""
    name = os.path.splitext(os.path.basename(img_path))[0]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) detect markers
    corners, ids = detect_aruco(gray, dict_name)
    if ids is None or len(corners) == 0:
        print(f"[WARN] {name}: no ArUco markers detected")
        return

    # Draw detections
    det_vis = img.copy()
    cv2.aruco.drawDetectedMarkers(det_vis, corners, ids)

    # 2) build mask from marker corners
    mask, hull = corners_to_mask(img.shape, corners, hull_mode="convex")
    if mask is None:
        print(f"[WARN] {name}: could not form mask")
        return

    # 3) refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    mask_ref = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 4) overlay visualization
    overlay = img.copy()
    cv2.drawContours(overlay, [hull.astype(np.int32)], -1, (0, 0, 255), 3)
    blend = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Save outputs
    ensure_dir(out_dir)
    cv2.imwrite(os.path.join(out_dir, f"{name}_aruco_detected.png"), det_vis)
    cv2.imwrite(os.path.join(out_dir, f"{name}_aruco_mask.png"), mask_ref)
    cv2.imwrite(os.path.join(out_dir, f"{name}_aruco_boundary.png"), blend)
    print(f"[OK] {name}: {len(ids)} markers → mask saved")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Input folder with images")
    ap.add_argument("--out_dir", default="data/outputs/aruco_segment", help="Output folder")
    ap.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary name")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    for img_path in list_images(args.in_dir):
        process_image(img_path, args.out_dir, dict_name=args.dict)

if __name__ == "__main__":
    main()

def segment_with_aruco(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(gray)
    result = img.copy()
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(result, corners, ids)
        pts = np.concatenate(corners).reshape(-1, 2).astype(np.int32)
        cv2.polylines(result, [pts], True, (0, 255, 0), 2)
    return result
