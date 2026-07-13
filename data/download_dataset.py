import argparse
import json
import os
import shutil
from pathlib import Path

import yaml


DEFAULT_CLASS_MAP = {
    "dent": "dent",
    "scratch": "scratch",
    "crack": "crack/shatter",
    "shatter": "crack/shatter",
    "crack/shatter": "crack/shatter",
    "bumper_damage": "bumper_damage",
    "broken_lamp": "broken_lamp",
    "lamp": "broken_lamp",
}

DEFAULT_SEVERITY_MAP = {
    "dent": "Moderate",
    "damage roof": "Severe",
    "damage vehicle back": "Moderate",
    "damage vehicle front": "Moderate",
    "damaged bumper": "Moderate",
    "damaged door": "Moderate",
    "damaged headlight": "Moderate",
    "damaged hood": "Moderate",
    "damaged mirror": "Minor",
    "damaged wind shield": "Severe",
    "damaged window": "Severe",
    "images": "Minor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download or register a vehicle damage dataset."
    )
    parser.add_argument(
        "--task",
        choices=["detection", "severity"],
        default="detection",
        help="Which dataset flow to prepare.",
    )
    parser.add_argument(
        "--source",
        choices=["local", "roboflow"],
        default="local",
        help="Dataset source. Use 'roboflow' when ROBOFLOW_API_KEY is available.",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        help="Existing dataset directory.",
    )
    parser.add_argument("--workspace", help="Roboflow workspace slug.")
    parser.add_argument("--project", help="Roboflow project slug.")
    parser.add_argument("--version", type=int, help="Roboflow dataset version.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where the dataset should be stored.",
    )
    parser.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("data/dataset.yaml"),
        help="Path for the normalized YOLO dataset config.",
    )
    parser.add_argument(
        "--severity-dir",
        type=Path,
        default=Path("data/severity"),
        help="Target ImageFolder directory for normalized severity classes.",
    )
    parser.add_argument(
        "--severity-mapping-json",
        type=Path,
        default=Path("data/severity_mapping.json"),
        help="Where to write the discovered severity class mapping.",
    )
    return parser.parse_args()


def download_roboflow_dataset(args: argparse.Namespace) -> Path:
    try:
        from roboflow import Roboflow
    except ImportError as exc:
        raise RuntimeError(
            "roboflow is not installed. Run 'pip install -r requirements.txt' first."
        ) from exc

    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required for --source roboflow.")
    if not args.workspace or not args.project or not args.version:
        raise RuntimeError(
            "--workspace, --project, and --version are required for --source roboflow."
        )

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(args.workspace).project(args.project)
    dataset_format = "yolov8" if args.task == "detection" else "folder"
    dataset = project.version(args.version).download(dataset_format, location=str(args.output_dir))
    return Path(dataset.location)


def register_local_dataset(local_dir: Path, output_dir: Path) -> Path:
    if not local_dir.exists():
        raise FileNotFoundError(f"Local dataset not found: {local_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / local_dir.name
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(local_dir, target)
    return target


def find_source_yaml(dataset_dir: Path) -> Path | None:
    for candidate in dataset_dir.rglob("data.yaml"):
        return candidate
    for candidate in dataset_dir.rglob("dataset.yaml"):
        return candidate
    return None


def normalize_names(names: list[str] | dict[int, str]) -> list[str]:
    if isinstance(names, dict):
        ordered = [names[index] for index in sorted(int(key) for key in names.keys())]
    else:
        ordered = names
    normalized: list[str] = []
    for name in ordered:
        key = str(name).strip().lower().replace(" ", "_")
        normalized.append(DEFAULT_CLASS_MAP.get(key, key))
    return normalized


def write_severity_mapping(names: list[str], destination: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name in names:
        normalized = name.strip().lower().replace("_", " ")
        mapping[name] = DEFAULT_SEVERITY_MAP.get(normalized, "Moderate")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(mapping, indent=2))
    return mapping


def write_dataset_yaml(dataset_root: Path, destination: Path) -> None:
    source_yaml = find_source_yaml(dataset_root)
    if source_yaml is None:
        names = ["dent", "scratch", "crack/shatter", "bumper_damage", "broken_lamp"]
    else:
        config = yaml.safe_load(source_yaml.read_text())
        names = normalize_names(config.get("names", []))

    dataset_config = {
        "path": str(dataset_root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(dataset_config, sort_keys=False))


def write_metadata(dataset_root: Path, source: str, task: str, version: int | None) -> None:
    metadata = {
        "dataset_root": str(dataset_root.resolve()),
        "source": source,
        "task": task,
        "version": version,
        "notes": (
            "Confirm license, actual sample counts, and class balance before "
            "reporting any training metrics."
        ),
    }
    metadata_path = Path("data") / "dataset_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))


def find_classification_root(dataset_root: Path) -> Path:
    for child in dataset_root.iterdir():
        if child.is_dir() and child.name.lower() in {"train", "valid", "test"}:
            split_dirs = [entry for entry in child.iterdir() if entry.is_dir()]
            if split_dirs:
                return child
    return dataset_root


def normalize_severity_dataset(dataset_root: Path, destination: Path, mapping_path: Path) -> None:
    classification_root = find_classification_root(dataset_root)
    split = "train" if (classification_root / "train").exists() else ""
    source_root = classification_root / split if split else classification_root

    class_dirs = [item for item in source_root.iterdir() if item.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No severity class folders found in: {source_root}")

    mapping = write_severity_mapping([item.name for item in class_dirs], mapping_path)
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    for class_dir in class_dirs:
        bucket = mapping[class_dir.name]
        target_dir = destination / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        for image_path in class_dir.iterdir():
            if image_path.is_file():
                shutil.copy2(image_path, target_dir / f"{class_dir.name}_{image_path.name}")


def main() -> None:
    args = parse_args()

    if args.source == "roboflow":
        dataset_root = download_roboflow_dataset(args)
    else:
        if not args.local_dir:
            raise RuntimeError("--local-dir is required for --source local.")
        dataset_root = register_local_dataset(args.local_dir, args.output_dir)

    if args.task == "detection":
        write_dataset_yaml(dataset_root, args.dataset_yaml)
        print(f"Dataset ready at: {dataset_root}")
        print(f"YOLO config written to: {args.dataset_yaml}")
        print("Next step: python3 models/train_detector.py --data data/dataset.yaml")
    else:
        normalize_severity_dataset(dataset_root, args.severity_dir, args.severity_mapping_json)
        print(f"Severity dataset ready at: {args.severity_dir}")
        print(f"Severity mapping written to: {args.severity_mapping_json}")
        print("Next step: python3 models/train_classifier.py --data-dir data/severity")

    write_metadata(dataset_root, args.source, args.task, args.version)


if __name__ == "__main__":
    main()
