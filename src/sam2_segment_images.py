#!/usr/bin/env python3
import sys, os
sys.path.append("src")

import streamlit as st
import cv2
import numpy as np
from pathlib import Path


# ==========================================================
# Utility
# ==========================================================
def load_image(path):
    if path is None or not Path(path).exists():
        return None
    img = cv2.imread(str(path))
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ==========================================================
# Directories (MATCH YOUR IMPLEMENTATION)
# ==========================================================
DIR_RAW = Path("data/raw")
DIR_RAW_ARUCO = Path("data/raw_aruco")

DIR_Q1 = Path("data/outputs/gradlog")
DIR_Q2_EDGES = Path("data/outputs/edge_keypoints")
DIR_Q2_CORNERS = Path("data/outputs/corner_keypoints")
DIR_Q3 = Path("data/outputs/boundary_extract")
DIR_Q4 = Path("data/outputs/aruco_segment")

# CORRECT SAM2 PATH + FILENAME FORMAT
DIR_Q5 = Path("data/outputs/sam2_overlays")


# ==========================================================
# STREAMLIT UI
# ==========================================================
st.set_page_config(page_title="CSc 8830 – Assignment 3", layout="wide")
st.title("CSc 8830 – Computer Vision Assignment 3")

tabs = st.tabs([
    "Q1 – Gradient & LoG",
    "Q2 – Edge and Corner Keypoints",
    "Q3 – Boundary Extraction",
    "Q4 – ArUco + SAM2 Segmentation"
])

# ==========================================================
# Q1 – Gradient & LoG  (UNCHANGED)
# ==========================================================
with tabs[0]:
    st.header("Q1 – Gradient Magnitude, Orientation & Laplacian of Gaussian")

    files = sorted(DIR_Q1.glob("*_grad_mag.png"))
    if not files:
        st.warning(f"No Q1 output files found in {DIR_Q1}.")
    else:
        file = st.selectbox("Select image", files, key="q1")
        stem = file.name.replace("_grad_mag.png", "")

        grad_mag = load_image(file)
        grad_ang = load_image(DIR_Q1 / f"{stem}_grad_ang.png")
        log_img = load_image(DIR_Q1 / f"{stem}_log.png")

        original = load_image(DIR_RAW / f"{stem}.jpg") or load_image(DIR_RAW / f"{stem}.png")
        if original is None:
            st.error(f"Original image for {stem} not found.")
            st.stop()

        col1, col2, col3, col4 = st.columns(4)
        with col1: st.image(original, caption="Original Image", use_container_width=True)
        with col2: st.image(grad_mag, caption="Gradient Magnitude", use_container_width=True)
        with col3: st.image(grad_ang, caption="Gradient Angle (HSV)", use_container_width=True)
        with col4: st.image(log_img, caption="Laplacian of Gaussian", use_container_width=True)


# ==========================================================
# Q2 – Edge Keypoints + Corners  (UNCHANGED)
# ==========================================================
with tabs[1]:
    st.header("Q2 – Edge and Corner Keypoints")

    edge_outputs = sorted(DIR_Q2_EDGES.glob("*_grad_edges_overlay.png"))
    if not edge_outputs:
        st.warning(f"No edge keypoint files found in {DIR_Q2_EDGES}.")
    else:
        file = st.selectbox("Select image", edge_outputs, key="q2")
        stem = file.name.replace("_grad_edges_overlay.png", "")

        raw = load_image(DIR_RAW / f"{stem}.jpg") or load_image(DIR_RAW / f"{stem}.png")

        edges = load_image(file)
        corners = load_image(DIR_Q2_CORNERS / f"{stem}_corners.png")

        col1, col2, col3 = st.columns(3)
        with col1: st.image(raw, caption="Original Image", use_container_width=True)
        with col2: st.image(edges, caption="Gradient-Based Edge Keypoints", use_container_width=True)
        with col3: st.image(corners, caption="Shi–Tomasi Corners", use_container_width=True)


# ==========================================================
# Q3 – Boundary Extraction (UNCHANGED)
# ==========================================================
with tabs[2]:
    st.header("Q3 – Boundary Extraction (Classical CV)")

    boundary_files = sorted(DIR_Q3.glob("*_boundary.png"))
    if not boundary_files:
        st.warning(f"No boundary extraction outputs found in {DIR_Q3}.")
    else:
        selected = st.selectbox("Select result", boundary_files, key="q3")
        stem = selected.name.replace("_boundary.png", "")

        orig = load_image(DIR_RAW / f"{stem}.jpg") or load_image(DIR_RAW / f"{stem}.png")
        boundary = load_image(selected)

        col1, col2 = st.columns(2)
        with col1: st.image(orig, caption="Original Image", use_container_width=True)
        with col2: st.image(boundary, caption="Extracted Boundary", use_container_width=True)


# ==========================================================
# Q4 – COMBINED ArUco Marker Segmentation + SAM2 Deep Segmentation
# ==========================================================
with tabs[3]:
    st.header("Q4 – ArUco Marker Segmentation + SAM2 Comparison")

    q4_files = sorted(DIR_Q4.glob("*_aruco_boundary.png"))
    if not q4_files:
        st.warning(f"No Q4 outputs found in {DIR_Q4}.")
    else:
        selected = st.selectbox("Select image", q4_files, key="q4")
        stem = selected.name.replace("_aruco_boundary.png", "")

        # Load original
        orig = load_image(DIR_RAW_ARUCO / f"{stem}.jpg") or load_image(DIR_RAW_ARUCO / f"{stem}.png")

        # Load ArUco results
        aruco_boundary = load_image(DIR_Q4 / f"{stem}_aruco_boundary.png")
        aruco_mask     = load_image(DIR_Q4 / f"{stem}_aruco_mask.png")

        # ------------------------------------------------------------
        # FIXED SAM2 filename format:
        #     "1.jpg_sam2_overlay.png"
        # ------------------------------------------------------------
        sam2_filename = f"{stem}.jpg_sam2_overlay.png"
        sam2_path = DIR_Q5 / sam2_filename

        sam2_overlay = load_image(sam2_path) if sam2_path.exists() else None

        # Layout: 2 × 2 grid
        col1, col2 = st.columns(2)
        col3, col4 = st.columns(2)

        with col1:
            st.image(orig, caption="Original Image", use_container_width=True)

        with col2:
            st.image(aruco_boundary, caption="ArUco Boundary Overlay", use_container_width=True)

        with col3:
            st.image(aruco_mask, caption="Mask from Marker Hull", use_container_width=True)

        with col4:
            if sam2_overlay is None:
                st.error(f"SAM2 output NOT FOUND:\n{sam2_path}")
            else:
                st.image(sam2_overlay, caption="SAM2 Segmentation Overlay", use_container_width=True)
