# CityLens Analytics & Feature Set

> **CityLens AI Hackathon 2026** submission focusing on urban municipal monitoring via CCTV feeds.

## Overview
This repository contains our models developed during the intensive 48-hour CityLens AI Hackathon. We utilized YOLO26 architectures to solve three priority problem statements regarding urban cleanliness and traffic compliance:
1. **Encroachment Detection:** Identifying obstacles and size estimation in public spaces.
2. **Littering:** Cleanliness assessment and garbage detection on city roads.
3. **Illegal Parking:** Violation detection, average overstay duration, and parking occupancy tracking.

## Repository Structure
Due to GitHub's 100MB file limit, the compiled `.pt` model weights are excluded. The repository includes our inference scripts, evaluation metrics, and architecture details.

* **`Encroachment/`**: Utilizes an NMS-free YOLO26-Small backbone for high-speed edge inference.
* **`Littering/`**: Utilizes a high-resolution YOLO26-Large architecture to detect micro-objects.
* **`Illegal_Parking/`**: Utilizes YOLO26-Large to classify available, full, legal, and illegal parking states.

## Performance Metrics
* **Encroachment:** 92.40% mAP@0.5 | 96.07% Precision
* **Littering:** 71.66% mAP@0.5 | 61.21% Precision
* **Illegal Parking:** 64.66% mAP@0.5 | 72.01% Precision

## Environment Setup
To replicate the environment and run the inference scripts, ensure you have Python 3.8 or higher installed.

1. Clone this repository and navigate to the specific model folder you want to test (e.g., `Encroachment`):
   ```bash
   cd Encroachment
   ```
2. Install the required dependencies using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run Inference
Each folder operates independently. To run predictions on new test images, you must first place the corresponding trained `best.pt` weights file inside the folder.

Run the inference script from within the specific model's directory:
```bash
python inference.py --source DATA/test_images/input_frames --out DATA/test_images/output_frames
```

**Expected Output:**
* The script will process all images in the `--source` folder.
* Annotated frames (with bounding boxes drawn) will be saved to the `--out` folder.
* A `predictions.csv` file will be dynamically generated, formatted with normalized bounding box coordinates (`x_center`, `y_center`, `width`, `height`).
