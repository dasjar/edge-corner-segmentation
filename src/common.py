import os
import cv2 as cv
import numpy as np

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def list_images(in_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
    """Return sorted list of image file paths inside a directory."""
    files = [os.path.join(in_dir, f) for f in sorted(os.listdir(in_dir))]
    return [f for f in files if os.path.splitext(f)[1].lower() in exts]

def read_image(path: str) -> np.ndarray:
    img = cv.imread(path, cv.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img

def write_image(path: str, img: np.ndarray) -> None:
    ensure_dir(os.path.dirname(path))
    cv.imwrite(path, img)

def to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

def normalize_8u(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    mn, mx = np.min(x), np.max(x)
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    y = (x - mn) / (mx - mn) * 255.0
    return y.astype(np.uint8)

def gradient_mag_angle(gray: np.ndarray, ksize: int = 3):
    gx = cv.Sobel(gray, cv.CV_32F, 1, 0, ksize=ksize)
    gy = cv.Sobel(gray, cv.CV_32F, 0, 1, ksize=ksize)
    mag, ang = cv.cartToPolar(gx, gy, angleInDegrees=True)
    return mag, ang

def laplacian_of_gaussian(gray: np.ndarray, sigma: float = 1.2):
    blur = cv.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    log = cv.Laplacian(blur, cv.CV_32F, ksize=3)
    return log

def angle_to_hsv_bgr(ang_deg: np.ndarray, mag: np.ndarray | None = None) -> np.ndarray:
    """Visualize gradient angles as color (Hue=angle, Value=mag)."""
    h = (ang_deg / 360.0 * 179.0).astype(np.uint8)
    s = np.full_like(h, 255, dtype=np.uint8)
    if mag is None:
        v = np.full_like(h, 255, dtype=np.uint8)
    else:
        v = normalize_8u(mag)
    hsv = cv.merge([h, s, v])
    return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
