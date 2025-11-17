Edge–Corner Detection & Image Segmentation
CSc 8830 – Computer Vision (Assignment 3)

Author: Victor Solomon
Institution: Georgia State University
Instructor: Dr. Ashwin Ashok

Overview

This repository contains the full implementation, analysis pipeline, and interactive web application for Assignment 3: Edge/Corner Detection and Object Segmentation in CSc 8830: Computer Vision.
The project integrates classical image processing, feature detection, ArUco-based segmentation, and modern deep segmentation (SAM2) into a unified and reproducible system.

All tasks are implemented in Python using OpenCV, and a fully interactive demonstration is provided through a Streamlit web application, enabling real-time visualization of all steps.

Project Highlights

This project demonstrates:

Precise computation of image gradients, gradient orientation, and Laplacian of Gaussian (LoG)

Classical edge keypoint extraction and Shi–Tomasi corner detection

Exact object boundary extraction using smoothing, filtering, Canny edges, and contour analysis

Robust non-rectangular object segmentation using ArUco markers

Direct comparison with Meta’s SAM2 segmentation model

A complete research-grade visualization interface using Streamlit

Directory Structure
edge-corner-segmentation/
│
├── app.py                         # Streamlit interactive application
├── src/                           # All question-specific code modules
│   ├── grad_and_log.py
│   ├── edge_keypoints.py
│   ├── corner_keypoints.py
│   ├── boundary_extract.py
│   ├── aruco_segment.py
│   └── common.py
│
├── data/
│   ├── raw/                       # Raw images for Q1, Q2, Q3
│   ├── raw_aruco/                 # Images with ArUco markers for Q4 + Q5
│   └── outputs/
│       ├── gradlog/               # Q1 outputs
│       ├── edge_keypoints/        # Q2 edge outputs
│       ├── corner_keypoints/      # Q2 corner outputs
│       ├── boundary_extract/       # Q3 outputs
│       ├── aruco_segment/         # Q4 outputs
│       └── sam2_overlays/         # Q5 SAM2 overlays
│
├── requirements.txt               # All project dependencies
└── README.md                      # This document

Assignment Components
Q1 — Gradient Magnitude, Gradient Orientation & Laplacian of Gaussian

Implements:

Sobel-based gradient computation

Magnitude and angle visualization

HSV-encoded orientation for intuitive viewing

Laplacian of Gaussian (LoG) for blob/edge localization

Side-by-side comparison of gradient vs LoG features

Outputs saved to: data/outputs/gradlog/
Example files:

1_grad_mag.png
1_grad_ang.png
1_log.png

Q2 — Edge Keypoints & Shi–Tomasi Corner Detection

Implements:

Gradient magnitude thresholding (lecture-based edge keypoint extraction)

Shi–Tomasi minimum-eigenvalue corner detection

Sub-pixel corner refinement

Overlay visualization for both feature types

Outputs saved to:

data/outputs/edge_keypoints/
data/outputs/corner_keypoints/


Example files:

1_grad_edges_overlay.png
1_corners.png

Q3 — Classical Boundary Extraction

A classical pipeline combining:

Histogram equalization

Laplacian sharpening

Gaussian smoothing

Canny edge detection

Morphological closing

Contour filtering + scoring

Clean boundary mask extraction

Outputs in:

data/outputs/boundary_extract/


Example files:

1_boundary.png

Q4 — ArUco Marker–Based Segmentation

Robust non-rectangular object segmentation using only marker geometry:

Steps:

Detect markers using OpenCV’s cv2.aruco

Aggregate corner coordinates

Compute a convex hull mask

Apply morphological refinement

Overlay detected boundary on the original image

Outputs:

1_aruco_boundary.png
1_aruco_mask.png

Q5 — SAM2 Deep Segmentation (Comparison Only)

The project includes:

An evaluation script for SAM2

Bounding-box-guided segmentation using ArUco markers

SAM2 overlays for direct comparison with classical Q4 segmentation

Outputs:

1.jpg_sam2_overlay.png


This allows a real-world comparison of classical geometric segmentation vs. state-of-the-art foundation models.

Interactive Web Application

The full system is viewable through a Streamlit application:

streamlit run app.py


The UI provides:

Clean tab-based navigation per assignment question

Research-level visualization layouts

Automated loading of all precomputed outputs

Side-by-side comparisons for interpretability

Full integration of ArUco + SAM2 segmentation for Q4

The app is designed to match the structure of your assignment and provide a polished demonstration suitable for both academic submission and presentation.

Requirements

Install dependencies:

pip install -r requirements.txt


Must include (at minimum):

streamlit
opencv-python-headless
numpy
Pillow


If using SAM2 locally:

torch
torchvision
hydra-core

Running the Project
1. Generate outputs (Q1–Q4)

Each module inside src/ can be run independently, e.g.:

python src/grad_and_log.py
python src/edge_keypoints.py
python src/corner_keypoints.py
python src/boundary_extract.py
python src/aruco_segment.py

2. Run SAM2 segmentation (optional)
python sam2_segment_images.py --in_dir data/raw_aruco --out_dir data/outputs/sam2_overlays

3. Launch the app
streamlit run app.py

Deployment (Streamlit Cloud)

You can deploy the app online through:

https://share.streamlit.io

Select your GitHub repo

Set entrypoint → app.py

Deploy

Works automatically as long as data/ and requirements.txt are included.

Academic Integrity & Citations

If using ArUco, SAM2, or external resources, please cite appropriately.
Example citations:

OpenCV Contributors (2023). OpenCV Library.

Meta AI Research (2024). Segment Anything Model 2 (SAM2).

Conclusion

This project demonstrates a complete pipeline—from classical gradient-based feature extraction to modern foundation-model segmentation—wrapped inside a robust, interactive visualization system.
It reflects applied competency in:

Classical computer vision

Geometric segmentation

Marker-based spatial reasoning

Deep learning model integration

Real-time system design