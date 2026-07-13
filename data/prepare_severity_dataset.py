import argparse
import csv
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare severity classifier crops from detection annotations."
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        required=True,
        help=(
            "CSV file with columns: image_name,x1,y1,x2,y2,severity. "
            "Severity must be Minor, Moderate, or Severe."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/severity"),
        help="Output directory formatted for torchvision.datasets.ImageFolder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.labels_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            severity = row["severity"].strip()
            image_path = args.images_dir / row["image_name"]
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

            crop_dir = args.output_dir / severity
            crop_dir.mkdir(parents=True, exist_ok=True)

            image = Image.open(image_path).convert("RGB")
            bbox = tuple(int(float(row[key])) for key in ("x1", "y1", "x2", "y2"))
            crop = image.crop(bbox)
            crop.save(crop_dir / f"{image_path.stem}_{index}.png")

    print(f"Severity dataset prepared at: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
