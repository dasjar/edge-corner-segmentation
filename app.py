#!/usr/bin/env python3
import sys, os
sys.path.append("/data/users2/vsolomon3/csc8830-cv-assignment3/src")

import streamlit as st
import cv2
import numpy as np
from pathlib import Path

from src.grad_and_log import compute_grad_and_log
from src.edge_keypoints import detect_edges
from src.corner_keypoints import detect_corners
from src.boundary_extract import extract_boundary
from src.aruco_segment import segment_with_aruco



# ==========================================================
# Streamlit UI setup
# ==========================================================
st.set_page_config(page_title="CSc 8830 — Assignment 3", layout="wide")
st.title("CSc 8830 — Computer Vision Assignment 3")
st.write("**Parts 1–4: Gradients, Keypoints, Boundaries, and ArUco Segmentation**")

# Dataset directory
dataset_dir = Path("data/raw_aruco")
images = sorted([p for p in dataset_dir.glob("*.jpg")])

if not images:
    st.error("No images found in data/raw_aruco/. Please upload your dataset first.")
    st.stop()

# Select image
img_names = [p.name for p in images]
choice = st.selectbox("Select an image:", img_names)
img_path = str(dataset_dir / choice)
img = cv2.imread(img_path)

col1, col2 = st.columns(2)
with col1:
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original Image", use_column_width=True)

# Choose which part to visualize
task = st.radio(
    "Choose a part to visualize:",
    [
        "1️⃣ Gradient + Laplacian of Gaussian",
        "2️⃣ Edge + Corner Keypoints",
        "3️⃣ Boundary Extraction",
        "4️⃣ ArUco Marker Segmentation"
    ],
    horizontal=False
)

# ==========================================================
# Run the selected task
# ==========================================================
if task.startswith("1"):
    mag, ang, log = compute_grad_and_log(img)
    st.image([mag, ang, log],
             caption=["Gradient Magnitude", "Gradient Angle", "Laplacian of Gaussian"],
             use_column_width=True)

elif task.startswith("2"):
    edges = detect_edges(img)
    corners = detect_corners(img)
    st.image([edges, corners],
             caption=["Detected Edges", "Detected Corners"],
             use_column_width=True)

elif task.startswith("3"):
    boundary_img = extract_boundary(img)
    st.image(boundary_img, caption="Extracted Object Boundary", use_column_width=True)

elif task.startswith("4"):
    aruco_result = segment_with_aruco(img)
    st.image(aruco_result, caption="ArUco Marker Segmentation", use_column_width=True)
