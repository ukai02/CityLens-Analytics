"""
inference.py — reproduces predictions.csv and output_frames/ from this
submission alone. Needs only best.pt + input images.

Usage:
    python inference.py --source DATA/test_images/input_frames --out DATA/test_images/output_frames
"""
import argparse, csv, cv2
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="CORE/best.pt")
    p.add_argument("--source", default="DATA/test_images/input_frames")
    p.add_argument("--out", default="DATA/test_images/output_frames")
    p.add_argument("--csv", default="CORE/predictions.csv")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.25)
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
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xc, yc, bw, bh = box.xywhn[0].tolist()
            rows.append({
                "image": img_path.name, "class_id": cls_id,
                "class_name": names[cls_id], "confidence": round(conf, 4),
                "xmin": round(x1, 2), "ymin": round(y1, 2),
                "xmax": round(x2, 2), "ymax": round(y2, 2),
                "x_center_norm": round(xc, 6), "y_center_norm": round(yc, 6),
                "width_norm": round(bw, 6), "height_norm": round(bh, 6),
            })

    if rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
        print(f"Saved {len(rows)} detections to {args.csv}")
    print(f"Annotated frames saved to {args.out}")

if __name__ == "__main__":
    main()
