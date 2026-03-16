# Image Stitching Project

A computer vision project to learn and implement image stitching (panorama generation) using feature detection, matching, homography estimation and blending.

## Project Structure

image-stitching/
│
├── src/
│   └── stitching/
│       ├── feature.py
│       ├── matcher.py
│       ├── homography.py
│       ├── blender.py
│       └── pipeline.py
│
├── data/
│   ├── sample/
│   └── pano/
│
├── notebooks/
│
├── outputs/
│
├── tests/
│
├── main.py
├── requirements.txt
└── README.md

## Setup

Create virtual environment:

python -m venv venv

Activate environment:

### Windows
venv\Scripts\activate

### macOS / Linux
source venv/bin/activate


Install dependencies:
pip install -r requirements.txt

## Run Stitching Pipeline
python main.py

Output panorama will be saved inside:

outputs/

## Pipeline Overview

Image stitching pipeline consists of:

1. Feature Detection (ORB / SIFT)
2. Feature Matching
3. Homography Estimation (RANSAC)
4. Image Warping
5. Blending / Seam correction

---
## Learning Goals

* Understand geometric transformations
* Implement homography estimation
* Compare feature detectors (ORB vs SIFT)
* Build scalable multi-image panorama pipeline
* Optimize stitching quality and performance

---
## Future Improvements

* Multi-image panorama stitching
* Exposure compensation
* Seam finding
* GPU acceleration
* CLI interface
* Dataset benchmarking