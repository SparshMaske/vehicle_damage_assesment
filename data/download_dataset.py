from pathlib import Path


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    message = """
Dataset download is intentionally not automated in this workspace because network
access and dataset licensing vary by environment.

Recommended steps:
1. Download a public vehicle damage detection dataset in YOLO or COCO format.
2. Place it under data/raw/.
3. Convert or normalize labels if needed.
4. Export train/val splits for YOLOv8 training.
5. Derive severity labels for cropped damage regions if classifier training is required.
"""
    print(message.strip())
    print(f"Data directory: {data_dir}")


if __name__ == "__main__":
    main()
