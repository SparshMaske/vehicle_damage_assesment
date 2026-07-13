from pathlib import Path


def main() -> None:
    weights_dir = Path(__file__).resolve().parent / "weights"
    print("Severity classifier training scaffold")
    print(f"Store trained weights in: {weights_dir}")
    print("Expected labels: Minor, Moderate, Severe")
    print("Use a transfer-learning pipeline based on ResNet18 or EfficientNet-B0.")


if __name__ == "__main__":
    main()
