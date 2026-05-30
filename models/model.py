from pathlib import Path
from ultralytics import YOLO

# Inference script

MODEL_PATH = Path(__file__).parents[1] / "runs/detect/train-9/weights/best.pt"

model = YOLO("yolo_FAV.pt")

results = model.predict(
    source=0,  # Logitech Brio 101 on /dev/video0
    show=True,
    device=0,
    half=False,  # Model quantized to FP16 precision
    conf=0.5,  # Confidence level
    iou=0.4,  # Intersection on union threshold
    stream=True,
)

# Identify localised objects based on labeled bounding boxes
for r in results:
    names = [r.names[int(c)] for c in r.boxes.cls]
    if names:
        print(names)
