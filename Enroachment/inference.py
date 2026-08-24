"""
inference.py — reproduces predictions.csv and output_frames/ from this submission.
Requires best.pt and input images.

Usage:
    python inference.py --source DATA/test_images/input_frames --out DATA/test_images/output_frames
"""
import argparse, csv, cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser(description="Run Encroachment Inference")
    p.add_argument("--weights", default="best.pt", help="Path to model weights")
    p.add_argument("--source", default="DATA/test_images/input_frames", help="Input directory")
    p.add_argument("--out", default="DATA/test_images/output_frames", help="Output directory")
    p.add_argument("--csv", default="predictions.csv", help="CSV output path")
    p.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    p.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    args = p.parse_args()

    model = YOLO(args.weights)
    names = model.names
    Path(args.out).mkdir(parents=True, exist_ok=True)

    rows = []
    for r in model.predict(source=args.source, imgsz=args.imgsz,
                            conf=args.conf, save=False, stream=True):
        img_path = Path(r.path)
        cv2.imwrite(str(Path(args.out) / img_path.name), r.plot())
        
        if r.boxes is None or len(r.boxes) == 0:
            continue
            
        for box in r.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            xc, yc, bw, bh = box.xywhn[0].tolist()
            
            rows.append({
                "frame": img_path.name, "class_id": cls_id,
                "class_name": names[cls_id], 
                "x_center": f"{xc:.4f}", "y_center": f"{yc:.4f}",
                "width": f"{bw:.4f}", "height": f"{bh:.4f}",
                "confidence": f"{conf:.4f}"
            })

    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {len(rows)} detections to {args.csv}")
    print(f"Annotated frames saved to {args.out}")

if __name__ == "__main__":
    main()