# Fine-tunes YOLOv11n on a produce freshness dataset.
# Expects data.yaml at the project root pointing to your annotated dataset.
# Training was run on a GCP Compute Engine VM (NVIDIA T4 GPU).

from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")

    print("Starting training of yolo11n...")

    results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=32,
        device=0,
        cache=True,
        save=True,
        compile=True,
    )


if __name__ == "__main__":
    main()
