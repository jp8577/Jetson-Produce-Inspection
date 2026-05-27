from ultralytics import YOLO


def main():
    # load pretrained model
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
