"""
No custom architecture used.
Base model: yolo26s.pt (Ultralytics YOLO26-Small, stock).
Modifications: 
- NMS-free architecture utilized for high-speed edge inference.
- Granular sub-classes flattened into a unified 'encroachment' class (Class 0).
"""