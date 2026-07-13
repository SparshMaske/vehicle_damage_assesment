from pathlib import Path


def main() -> None:
    weights_dir = Path(__file__).resolve().parent / "weights"
    print("YOLOv8 detector training scaffold")
    print(f"Store trained weights in: {weights_dir}")
    print("Suggested command after dataset preparation:")
    print(
        "yolo detect train data=data/dataset.yaml model=yolov8n.pt "
        "epochs=50 imgsz=640 project=models/runs name=detector"
    )


if __name__ == "__main__":
    main()
