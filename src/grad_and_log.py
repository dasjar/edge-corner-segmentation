import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import os
import argparse
import numpy as np
import cv2 as cv
from src.common import (
    list_images, read_image, write_image, to_gray,
    gradient_mag_angle, laplacian_of_gaussian, normalize_8u, angle_to_hsv_bgr,
    ensure_dir
)

def run(in_dir: str, out_dir: str, sobel_ksize: int = 3, log_sigma: float = 1.2):
    ensure_dir(out_dir)
    paths = list_images(in_dir)
    if not paths:
        print(f"[WARN] No images found in {in_dir}")
    for p in paths:
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        img = read_image(p)
        gray = to_gray(img)

        mag, ang = gradient_mag_angle(gray, ksize=sobel_ksize)
        mag_u8 = normalize_8u(mag)
        ang_vis = angle_to_hsv_bgr(ang, mag)

        log = laplacian_of_gaussian(gray, sigma=log_sigma)
        log_u8 = normalize_8u(np.abs(log))

        write_image(os.path.join(out_dir, f"{stem}_grad_mag.png"), mag_u8)
        write_image(os.path.join(out_dir, f"{stem}_grad_ang.png"), ang_vis)
        write_image(os.path.join(out_dir, f"{stem}_log.png"), log_u8)

        panel = np.hstack([img, cv.cvtColor(mag_u8, cv.COLOR_GRAY2BGR), log_u8[...,None].repeat(3,axis=2)])
        write_image(os.path.join(out_dir, f"{stem}_panel_grad_vs_log.png"), panel)
        print(f"[OK] {name}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="data/raw")
    ap.add_argument("--out_dir", default="data/outputs/gradlog")
    ap.add_argument("--sobel_ksize", type=int, default=3, choices=[1,3,5,7])
    ap.add_argument("--log_sigma", type=float, default=1.2)
    args = ap.parse_args()
    run(**vars(args))

def compute_grad_and_log(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)
    mag = cv.magnitude(gx, gy)
    ang = cv.phase(gx, gy, angleInDegrees=True)
    log = cv.Laplacian(cv.GaussianBlur(gray, (5,5), 0), cv.CV_64F)
    return [cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8),
            cv.normalize(ang, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8),
            cv.normalize(np.abs(log), None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)]
