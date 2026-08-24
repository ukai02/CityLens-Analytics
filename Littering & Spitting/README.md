# Parking CityLens — Model Submission

## Overview
Brief description of your approach.

## Model
- Base architecture: YOLO26-Large (yolo26l.pt, stock)
- Classes: available, full, illegal, legal
- Input size: 1024x1024

## Training methodology
- Offline augmentation (copy-paste for rare class, etc.)
- Hyperparameters used

## Results
See metrics.csv

## How to run
```bash
pip install -r requirements.txt
python inference.py --source DATA/test_images/input_frames
```
