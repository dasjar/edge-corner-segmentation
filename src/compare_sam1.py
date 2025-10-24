#!/usr/bin/env python3
# -------------------------------------------------------------
# compare_sam1.py — Stable, memory-safe SAM v1 version
# -------------------------------------------------------------
# Compares ArUco-based segmentation masks with SAM predictions
# -------------------------------------------------------------

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def compute_iou_and_dice(mask1, mask2):
    mask1 = mask1.astype(bool)
    mask2 = mask2.astype(bool)
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    dice = 2 * intersection / (mask1.sum() + mask2.sum()) if (mask1.sum() + mask2.sum()) > 0 else 0.0
    return iou, dice


def main(in_dir, mask_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---------------------------------------------------------
    # Load small SAM model (ViT-B)
    # ---------------------------------------------------------
    checkpoint = "weights/sam_vit_b_01ec64.pth"
    model_type = "vit_b"

    print("[INFO] Loading SAM v1 (ViT-B) model...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    print("[INFO] SAM model ready.\n")

    results = []
    images = sorted([f for f in os.listdir(in_dir) if f.lower().endswith((".jpg", ".png"))])

    for fname in tqdm(images):
        name, _ = os.path.splitext(fname)
        img_path = os.path.join(in_dir, fname)
        mask_path = os.path.join(mask_dir, f"{name}_aruco_mask.png")

        if not os.path.exists(mask_path):
            print(f"[WARN] No mask for {fname}, skipping.")
            continue

        image = cv2.imread(img_path)
        mask_gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask_gt = (mask_gt > 127).astype(np.uint8)

        orig_h, orig_w = image.shape[:2]

        # -----------------------------------------------------
        # Downscale large images for GPU memory safety
        # -----------------------------------------------------
        if max(orig_h, orig_w) > 1024:
            scale = 1024 / max(orig_h, orig_w)
            image_resized = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))
        else:
            image_resized = image.copy()

        # -----------------------------------------------------
        # SAM prediction
        # -----------------------------------------------------
        masks = mask_generator.generate(image_resized)
        if not masks:
            print(f"[WARN] SAM failed on {fname}")
            continue

        mask_sam = max(masks, key=lambda m: m["area"])["segmentation"].astype(np.uint8)
        # Upscale SAM mask back to original size before comparison
        mask_sam = cv2.resize(mask_sam, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # -----------------------------------------------------
        # Align ground-truth mask to same resolution
        # -----------------------------------------------------
        if mask_gt.shape != mask_sam.shape:
            mask_gt = cv2.resize(mask_gt, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

        # -----------------------------------------------------
        # Compute IoU/Dice
        # -----------------------------------------------------
        iou, dice = compute_iou_and_dice(mask_gt, mask_sam)
        results.append((iou, dice))
        print(f"[OK] {fname}: IoU={iou:.3f}, Dice={dice:.3f}")

        # -----------------------------------------------------
        # Visualization (safe size alignment)
        # -----------------------------------------------------
        overlay_gt = image.copy()
        overlay_sam = image.copy()

        overlay_gt = cv2.resize(overlay_gt, (orig_w, orig_h))
        overlay_sam = cv2.resize(overlay_sam, (orig_w, orig_h))

        overlay_gt[mask_gt > 0] = [0, 0, 255]    # Red = ground truth (ArUco)
        overlay_sam[mask_sam > 0] = [0, 255, 0]  # Green = SAM mask

        concat = np.concatenate((image, overlay_gt, overlay_sam), axis=1)
        cv2.imwrite(os.path.join(out_dir, f"{name}_compare.png"), concat)

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    if results:
        ious, dices = zip(*results)
        print(f"\n[SUMMARY] Mean IoU={np.mean(ious):.3f}, Mean Dice={np.mean(dices):.3f}")
    else:
        print("[SUMMARY] No valid results found.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare ArUco segmentation with SAM v1 automatic segmentation")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    main(args.in_dir, args.mask_dir, args.out_dir)
