# Dataset Notes

Use a real annotated vehicle damage dataset before relying on model outputs.

Suggested public starting points:

- Roboflow Universe `Car Damage Detection` by CAPSTONE:
  `https://universe.roboflow.com/capstone-nh0nc/car-damage-detection-t0g92`
- Roboflow Universe `Vehicle Damage Severity` by Car Damage Severity:
  `https://universe.roboflow.com/car-damage-severity/vehicle-damage-severity`
- Kaggle `Car Damage Detection` by `anujms`:
  `https://www.kaggle.com/datasets/anujms/car-damage-detection`

Recommended preparation:

1. Confirm the dataset license and whether commercial-style demonstration use is acceptable.
2. Normalize labels into the target damage classes: `dent`, `scratch`, `crack/shatter`, `bumper_damage`, `broken_lamp`.
3. Export YOLO or COCO annotations for detection training.
4. Derive region crops plus `Minor` / `Moderate` / `Severe` labels for classifier training.
5. Record actual dataset size, train/validation split, and class balance in project notes before reporting any metrics.

This repository ships no dataset, includes no trained weights, and claims no model performance numbers.

## Dataset and license notes

- `Car Damage Detection` by CAPSTONE
  - verified on July 13, 2026
  - public Universe page listed `3,226` images and `4` dataset versions
  - license shown on the public page: `CC BY 4.0`
- `Vehicle Damage Severity` by Car Damage Severity
  - verified on July 13, 2026
  - public Universe page listed `517` images and `3` dataset versions
  - license shown on the public page: `CC BY 4.0`
- `Car Damage Detection` by `anujms` on Kaggle
  - listed in the prompt as an optional damaged-vs-whole pre-filter dataset
  - license details should be verified directly in Kaggle after authentication before redistribution or public sharing

## Severity mapping used in this project

The Roboflow `vehicle-damage-severity` dataset publishes classification labels that do not match the project’s target `Minor` / `Moderate` / `Severe` routing schema directly. `data/download_dataset.py --task severity` writes `data/severity_mapping.json` and currently buckets labels as follows:

- `damage roof`, `damaged wind shield`, `damaged window` -> `Severe`
- `dent`, `damage vehicle back`, `damage vehicle front`, `damaged bumper`, `damaged door`, `damaged headlight`, `damaged hood` -> `Moderate`
- `damaged mirror`, `images` -> `Minor`

This mapping is an implementation choice for the demo pipeline and should be reviewed against actual label semantics before production use.

## Expected detector dataset layout

```text
your-yolo-dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Expected severity CSV layout

Use `data/prepare_severity_dataset.py` with a CSV shaped like:

```csv
image_name,x1,y1,x2,y2,severity
claim_001.jpg,42,55,188,140,Minor
claim_001.jpg,210,80,320,190,Moderate
```
