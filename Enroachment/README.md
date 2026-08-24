# Encroachment CityLens — Model Submission

## Overview
Our approach utilizes state-of-the-art object detection to identify urban encroachments (such as stalls, construction materials, and physical blockages) on roads. We framed this as a single-class detection problem to maximize the model's precision and recall specifically for this priority category.

## Model
- Base architecture: YOLO26-Small (yolo26s.pt, stock with NMS-free adaptations for edge devices)
- Classes: encroachment (Class 0)
- Input size: 640x640

## Training methodology
- Dataset logic: Flattened multiple granular sub-classes into a single unified 'encroachment' class to align with Hackathon KPIs.
- Augmentations: Mosaic (1.0), Horizontal Flip (0.5), HSV adjustments, and Brightness/Exposure shifts (-20% to +20%).
- Hyperparameters: 50 epochs, batch size 16.

## Results
See metrics.csv

## How to run
```bash
pip install -r requirements.txt
python inference.py --source DATA/test_images/input_frames --out DATA/test_images/output_frames